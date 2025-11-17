# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum
import numpy as np
import copy
import datasets
import itertools
import os
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq

B_INST, E_INST = "[INST]", "[/INST]"

def tokenize_dialog(dialog, tokenizer):
    dialog[1]['content'] = f"<<SYS>> {dialog[0]['content']} <</SYS>> {dialog[1]['content']}"
    dialog = dialog[1:]
    prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
    answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    #Add labels, convert prompt token to -100 in order to ignore in loss function
    labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }
    # input_ids = tokenizer.apply_chat_template(dialog,add_generation_prompt=False,return_tensors="pt")

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_custom_dataset(data_dir, tokenizer, split):
    # dataset = datasets.load_dataset("OpenAssistant/oasst1", split=split)
    dataset = datasets.load_dataset(data_dir, split=split)

    dataset = dataset.map(lambda sample: {
        "message_id": sample["message_id"],
        "parent_id": sample["parent_id"],
        "text": sample["text"],
        },
        batched=True,)

    nodes = {}

    messages = {}
    root_ids = []

    for data in dataset:
        if data["parent_id"]:
            nodes[data["parent_id"]] = nodes.get(data["parent_id"], []) + [data["message_id"]]
        else:
            root_ids.append(data["message_id"])
        messages[data["message_id"]]=data["text"]

    def follow(thread, current_id):
        thread = copy.copy(thread) + [messages[current_id]]
        if current_id in nodes:
            new_threads = []
            for next_id in nodes[current_id]:
                new_threads += follow(thread, next_id)
            return new_threads
        else:
            return [thread]

    def get_threads_from_root(root_id):
        all_threads = []
        thread = [messages[root_id]]
        for cid in nodes[root_id]:
            all_threads += follow(thread, cid)
        return all_threads

    dataset = dataset.filter(lambda x: x["message_id"] in root_ids)
    dataset = dataset.map(lambda x: {"thread": get_threads_from_root(x["message_id"])}, remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: {"thread": [i for row in x["thread"] for i in row]}, batched=True)

    def to_dialog(thread):
        dialog = []
        for i, content in enumerate(thread):
            dialog.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
            })
        return {"dialog": dialog}

    dataset = dataset.map(lambda x: to_dialog(x["thread"]), remove_columns=list(dataset.features))
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer), remove_columns=list(dataset.features))

    return dataset

def get_dialog_dataset(data_dir, tokenizer, split):
    dataset = datasets.load_dataset(data_dir, split=split).shuffle()
    dataset = dataset.map(lambda sample: {
        "dialog": sample["dialog"]
        },
        batched=True,)
    
    dataset = dataset.map(lambda x: tokenize_dialog(x["dialog"], tokenizer), remove_columns=list(dataset.features))

    return dataset


def encode_adapter_name(config, active_adapter_name):
    for i, layer_names in enumerate(config.adapter_names):
        for j, name in enumerate(layer_names):
            if name == active_adapter_name:
                return [i, j]
    return [-1, -1]
            
factor_dict = {
    "AGR":"Agreeableness",
    "CON":"Conscientiousness",
    "EXT":"Extraversion",
    "EMS":"Emotional Stability",
    "NEU":"Neuroticism",
    "OPE":"Openness",
}

PERSONALITY_FACTORS = {"AGR", "CON", "EXT", "NEU", "OPE"}
PROFESSIONS = {"Doctor", "Artist", "Programmer"}


def sanitize_active_adapters(adapter_list):
    sanitized = []
    seen_factors = set()
    profession_added = False
    for name in adapter_list:
        if name in PROFESSIONS:
            if not profession_added:
                sanitized.append(name)
                profession_added = True
        elif "_" in name:
            factor = name.split("_")[0]
            if factor in PERSONALITY_FACTORS and factor not in seen_factors:
                sanitized.append(name)
                seen_factors.add(factor)
        if len(sanitized) == 6:
            break
    return sanitized

def build_system_prompt(adapter_names):
    active_factor_polar = {}
    active_prof = None
    for adapter_name in adapter_names:
        if "_" in adapter_name:
            factor = adapter_name.split("_")[0]
            polar = adapter_name.split("_")[1]
            active_factor_polar[factor] = polar
        else:
            active_prof = adapter_name

    if active_prof is not None:
        start_prompt = f"You are a/an {active_prof.lower()}"
    else:
        start_prompt = "You are a human"
    if len(active_factor_polar)>0:
        start_prompt += " with "
        for factor, polar in active_factor_polar.items():
            start_prompt += f"{polar} {factor_dict[factor].lower()}, "
        start_prompt = start_prompt[:-2]
    if active_prof is not None and len(active_factor_polar)>0:
        start_prompt += ". Your responses should reflect these personality traits, as well as your profession, ensuring that the conversation feels natural and true to your character."
    elif active_prof is None and len(active_factor_polar)>0:
        start_prompt += ". Your responses should reflect these personality traits, ensuring that the conversation feels natural and true to your character."
    elif active_prof is not None and len(active_factor_polar)==0:
        start_prompt += ". Your responses should reflect your profession, ensuring that the conversation feels natural and true to your character."
    else:
        start_prompt += ". Your responses should ensure that the conversation feels natural and true to your character."
    # print(start_prompt)
    return start_prompt

def tokenize_moe_dialog(dialog, tokenizer, active_adapters, config):
    limited_adapters = sanitize_active_adapters(active_adapters)
    dialog[0]['content'] = f"<<SYS>> {build_system_prompt(limited_adapters)} <</SYS>> {dialog[0]['content']}"
    prompt_tokens = [tokenizer.encode(f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST}", add_special_tokens=False) for prompt in dialog[::2]]
    answer_tokens = [tokenizer.encode(f"{answer['content'].strip()} {tokenizer.eos_token}", add_special_tokens=False) for answer in dialog[1::2]]
    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))
    #Add labels, convert prompt token to -100 in order to ignore in loss function
    labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i,c in enumerate(dialog_tokens)]
    # breakpoint()
    # print(f"active_adapters {active_adapters}")

    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
        "active_adapters": [encode_adapter_name(config, active_adapter) for active_adapter in limited_adapters] + (6-len(limited_adapters)) * [[-1,-1]]
    }
    # print(tokenizer.decode(list(itertools.chain(*(t for t in dialog_tokens)))))
    # if len(combined_tokens["active_adapters"]) != 6:
    #     print(len(combined_tokens["active_adapters"]), active_adapters)
    # print(dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"])))

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_moe_dataset(data_dir, tokenizer, split, config):
    data_files = {
        "train": os.path.join(data_dir, "train.json"),
        "test": os.path.join(data_dir, "test.json"),
    }
    if split not in data_files:
        raise ValueError(f"Unsupported split {split}. Available splits: {list(data_files)}")
    dataset = datasets.load_dataset(
        "json",
        data_files=data_files,
        field=None,
        keep_in_memory=True,
    )[split].shuffle()
    dataset = dataset.map(lambda sample: {
        "dialog": sample["dialog"],
        "active_adapters": sample["active_adapters"],
        },
        batched=True,)
    
    dataset = dataset.map(lambda x: tokenize_moe_dialog(x["dialog"], tokenizer, x["active_adapters"], config), remove_columns=list(dataset.features))

    return dataset
    # import random
    # return random.sample(dataset, 100)

# 自定义DataCollator来处理额外的输入
class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                feature["active_adapters"] = feature["active_adapters"] + (6-len(feature["active_adapters"])) * [[-1, -1]]
                print(feature)
                # print(len(feature["active_adapters"]))
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        
        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
