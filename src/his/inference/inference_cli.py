from __future__ import annotations

import itertools
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import torch
from tqdm import tqdm, trange
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from his.inference.prompts import (
    direct_prompt,
    human_direct_prompt,
    human_question_prompt,
    question_prompt,
)
from his.models.moe_lora import MLoraConfig, MLoraModelForCausalLM
from his.utils.gpt import gpt_api, gpt_dialog_completion

B_INST, E_INST = "[INST]", "[/INST]"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_MODEL = os.getenv(
    "HIS_BASE_MODEL",
    "/remote-home/lbsun/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
)
DEFAULT_LORA_PATH = os.getenv(
    "HIS_LORA_WEIGHTS", str(PROJECT_ROOT / "models" / "1116_artist")
)
DEFAULT_SCALE_PATH = os.getenv(
    "HIS_SCALE_PATH", str(PROJECT_ROOT / "benchmark" / "personality_scale.json")
)
DEFAULT_SCALES_DIR = Path(
    os.getenv("HIS_SCALES_DIR", str(PROJECT_ROOT / "scales"))
)

PERSONALITY_FACTORS = {"AGR", "CON", "EXT", "NEU", "OPE"}
PROFESSIONS = {"Doctor", "Artist", "Programmer"}

human = True
use_factor_prompt = True
direct = False

factor = "OPEN"
polarity = "low"
scales_save_dir = "agent_dialog_finetune+factor_prompt"
checkpoint = 100

tokenizer: Optional[LlamaTokenizer] = None
model: Optional[MLoraModelForCausalLM] = None
adapter_config: Optional[MLoraConfig] = None
_MODEL_CACHE: Dict[Tuple[str, str], Tuple[LlamaTokenizer, MLoraModelForCausalLM, Optional[MLoraConfig]]] = {}


def _default_dialog_dir(
    scales_dir: str,
    persona_factor: str,
    persona_polarity: str,
    ckpt: Optional[int],
) -> Path:
    base = DEFAULT_SCALES_DIR / scales_dir
    if scales_dir == "agent_dialog_prompt":
        target = base / persona_factor / f"prompt_{persona_polarity}"
    elif ckpt is not None:
        target = base / persona_factor / f"{persona_polarity}_{ckpt}"
    else:
        target = base / "base"
    target.mkdir(parents=True, exist_ok=True)
    return target


dialog_save_dir = _default_dialog_dir(scales_save_dir, factor, polarity, checkpoint)
scale_path = DEFAULT_SCALE_PATH


def _sanitize_active_adapters(adapter_names: Optional[List[str]]) -> List[str]:
    cleaned: List[str] = []
    seen_factors = set()
    profession_added = False
    for name in adapter_names or []:
        if name in PROFESSIONS:
            if not profession_added:
                cleaned.append(name)
                profession_added = True
        elif "_" in name:
            family = name.split("_")[0]
            if family in PERSONALITY_FACTORS and family not in seen_factors:
                cleaned.append(name)
                seen_factors.add(family)
        if len(cleaned) == 6:
            break
    return cleaned


def _ensure_model(
    base_model: Optional[str],
    lora_path: Optional[str],
    cuda_devices: Optional[str],
) -> Tuple[LlamaTokenizer, MLoraModelForCausalLM, Optional[MLoraConfig]]:
    global tokenizer, model, adapter_config

    base_model = base_model or DEFAULT_BASE_MODEL
    lora_path = lora_path or DEFAULT_LORA_PATH
    cache_key = (base_model, lora_path)
    if cache_key in _MODEL_CACHE:
        tokenizer, model, adapter_config = _MODEL_CACHE[cache_key]
        return tokenizer, model, adapter_config

    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    tokenizer.padding_side = "left"

    base = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map="auto",
    )

    adapter_cfg: Optional[MLoraConfig] = None
    if lora_path:
        adapter_cfg = MLoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            adapter_names=[
                [
                    "AGR_high",
                    "AGR_low",
                    "CON_high",
                    "CON_low",
                    "EXT_high",
                    "EXT_low",
                    "NEU_high",
                    "NEU_low",
                    "OPE_high",
                    "OPE_low",
                ],
                ["Doctor", "Artist", "Programmer"],
            ],
            insert_mode="flat",
            sparse_adapter=False,
            token_dim=base.get_input_embeddings().weight.shape[1],
        )
        adapter_cfg.inference_mode = True
        base = MLoraModelForCausalLM(base, adapter_cfg)
        adapter_path = Path(lora_path) / "adapter_model.bin"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter weights not found at {adapter_path}")
        adapter_state = torch.load(adapter_path, map_location="cpu")
        missing = base.load_state_dict(adapter_state, strict=False)
        if missing.unexpected_keys:
            print(f"Unexpected adapter keys: {missing.unexpected_keys}")
        print(f"Loaded LoRA adapters from {adapter_path}")
    else:
        print("Running base model without adapters.")

    model = base
    adapter_config = adapter_cfg
    _MODEL_CACHE[cache_key] = (tokenizer, model, adapter_config)
    return tokenizer, model, adapter_config


def _encode_adapter_name(name: str) -> List[int]:
    if adapter_config is None:
        return [-1, -1]
    for i, layer in enumerate(adapter_config.adapter_names):
        for j, candidate in enumerate(layer):
            if candidate == name:
                return [i, j]
    return [-1, -1]


def _build_active_adapter_tensor(
    adapter_names: List[str] | List[List[str]] | None,
    pad_to: int = 6,
) -> torch.Tensor:
    if adapter_config is None:
        return torch.zeros((1, pad_to, 2), dtype=torch.long, device=model.device)
    if adapter_names is None or len(adapter_names) == 0:
        adapter_names = [["Artist"]]
    elif isinstance(adapter_names[0], str):  # type: ignore[index]
        adapter_names = [_sanitize_active_adapters(adapter_names)]  # type: ignore[list-item]

    rows: List[List[List[int]]] = []
    for names in adapter_names:  # type: ignore[assignment]
        names = _sanitize_active_adapters(names)
        indices = [_encode_adapter_name(name) for name in names]
        if pad_to > len(indices):
            indices.extend([[-1, -1]] * (pad_to - len(indices)))
        rows.append(indices)
    tensor = torch.tensor(rows, dtype=torch.long, device=model.device)
    return tensor


def evaluate(
    prompt: str = "",
    dialog: List[dict] | None = None,
    temperature: float = 1,
    top_p: float = 0.75,
    top_k: int = 40,
    num_beams: int = 1,
    max_new_tokens: int = 64,
    active_adapter_names: Optional[List[str]] = None,
    **kwargs,
):
    dialog = dialog or []
    if dialog and not prompt:
        prompts = [
            f"{tokenizer.bos_token} {B_INST} {(turn['content']).strip()} {E_INST}"
            for turn in dialog[::2]
        ]
        answers = [
            f"{resp['content'].strip()} {tokenizer.eos_token}"
            for resp in dialog[1::2]
        ] + [""]
        prompt = "".join(itertools.chain.from_iterable(zip(prompts, answers)))

    print(f"Prompt: {prompt}\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        do_sample=True,
        **kwargs,
    )

    gen_kwargs = {}
    if adapter_config is not None:
        active_tensor = _build_active_adapter_tensor(active_adapter_names)
        batch_size = input_ids.shape[0]
        if active_tensor.shape[0] != batch_size:
            if active_tensor.shape[0] == 1:
                active_tensor = active_tensor.repeat(batch_size, 1, 1)
            else:
                raise ValueError("Mismatch between adapter batch and input batch.")
        beam_size = generation_config.num_beams or 1
        if beam_size > 1:
            active_tensor = active_tensor.repeat_interleave(beam_size, dim=0)
        gen_kwargs["active_adapters"] = active_tensor
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        )

    s = generation_output.sequences[0]
    output = (
        tokenizer.decode(s)
        .replace("  ", " ")
        .replace(prompt, "")
        .replace("<s>", "")
        .replace("</s>", "")
    )
    return output


def main(
    mode: str = "multi",
    max_turns: int = 1,
    user_prompt: str = "你的职业是什么？",
    active_adapter_names: Optional[List[str]] = None,
    base_model: Optional[str] = None,
    lora_weights: Optional[str] = None,
    cuda_devices: Optional[str] = None,
):
    _ensure_model(base_model, lora_weights, cuda_devices)
    dialog: List[dict] = []
    turns = max(1, max_turns)
    for _ in range(turns):
        if mode == "single":
            prompt = f"<s> {B_INST}{user_prompt} {E_INST}"
            output = evaluate(prompt=prompt, active_adapter_names=active_adapter_names)
            print(f"Output: {output}\n")
        else:
            dialog.append({"role": "user", "content": user_prompt})
            output = evaluate(dialog=dialog, active_adapter_names=active_adapter_names)
            dialog.append({"role": "assistant", "content": output})
            print(f"Assistant: {output}\n")
    return dialog


def inference(
    save_path: Optional[str] = None,
    human_override: Optional[bool] = None,
    direct_override: Optional[bool] = None,
    base_model: Optional[str] = None,
    lora_weights: Optional[str] = None,
    cuda_devices: Optional[str] = None,
):
    _ensure_model(base_model, lora_weights, cuda_devices)
    with open(scale_path) as f:
        item_list = json.load(f)
    item_with_ans_list = []
    for item in tqdm(item_list):
        if direct_override is None:
            _direct = direct
        else:
            _direct = direct_override
        if human_override is None:
            _human = human
        else:
            _human = human_override
        if _direct:
            prompt = (
                direct_prompt(item["content"])
                if not _human
                else human_direct_prompt(item["content"])
            )
        else:
            prompt = (
                question_prompt(item["content"])
                if not _human
                else human_question_prompt(item["content"])
            )

        prompt = "<s> [INST]" + prompt + "[/INST]"
        output = evaluate(prompt=prompt)
        print(f"Output: {output}\n")
        ans_list = [s for s in re.split("[()]", output) if len(s) == 1]
        ans = ans_list[0] if len(ans_list) == 1 else None

        item["ans"] = ans
        item["output"] = output
        item_with_ans_list.append(item)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(item_with_ans_list, f)


def dialogue(
    save_path: str,
    human_override: Optional[bool] = None,
    base_model: Optional[str] = None,
    lora_weights: Optional[str] = None,
    cuda_devices: Optional[str] = None,
):
    _ensure_model(base_model, lora_weights, cuda_devices)
    with open(scale_path, "r") as f:
        item_list = json.load(f)
    item_with_ans_list = []
    _human = human if human_override is None else human_override
    for item in tqdm(item_list):
        dialog = []
        if _human:
            user_content = "Answer the following question as a human."
            dialog.append({"role": "user", "content": user_content})
            output = evaluate(dialog=dialog)
            dialog.append({"role": "assistant", "content": output})
        user_content = (
            f"{item['content']} (A) Very Inaccurate. (B) Moderately Inaccurate. "
            "(C) Neither Accurate Nor Inaccurate. (D) Moderately Accurate. "
            "(E) Very Accurate. Choose the most approprate one from (A)(B)(C)(D)(E) "
            "and give a detailed explanation."
        )
        dialog.append({"role": "user", "content": user_content})
        output = evaluate(dialog=dialog)
        dialog.append({"role": "assistant", "content": output})

        ans_list = [s for s in re.split("[()]", output) if len(s) == 1]
        for _ in range(3):
            if ans_list:
                break
            user_content = "So which one will you choose?"
            dialog.append({"role": "user", "content": user_content})
            output = evaluate(dialog=dialog)
            dialog.append({"role": "assistant", "content": output})
            ans_list = [s for s in re.split("[()]", output) if len(s) == 1]

        print(f"Output: {output}\n")
        ans = ans_list[0] if ans_list else "C"
        item["ans"] = ans
        item["output"] = dialog
        item_with_ans_list.append(item)

    with open(save_path, "w") as f:
        json.dump(item_with_ans_list, f)


factor_dict = {
    "AGR": "Agreeableness",
    "CON": "Conscientiousness",
    "EXT": "Extraversion",
    "NEU": "Neuroticism",
    "OPEN": "Openness",
}


def chatgpt(dialog_save_dir: Optional[str] = None):
    _ensure_model(None, None, None)
    with open(PROJECT_ROOT / "benchmark" / "personality_scale.json", "r") as f:
        question_list = json.load(f)
    with open(PROJECT_ROOT / "benchmark" / "profession_scale.json", "r") as f:
        item_list = json.load(f)
    assert len(question_list) == len(item_list)
    size = len(item_list)
    _human = human
    _use_factor_prompt = use_factor_prompt
    start_prompt = (
        f"Answer the following questions as a human with {polarity} {factor_dict[factor]}."
        if _human and _use_factor_prompt
        else ("Answer the following questions as a human. " if _human else "")
    )

    dialog_root = Path(dialog_save_dir or DEFAULT_SCALES_DIR / "chatgpt_sessions")
    dialog_root.mkdir(parents=True, exist_ok=True)

    for i in trange(size):
        dialog_save_path = dialog_root / f"{i}.json"
        if dialog_save_path.exists():
            continue

        item = item_list[i]
        statement = item["content"]
        question = question_list[i]["content"]

        dialog: List[dict] = []
        user_content = start_prompt + f"{question}"
        dialog.append({"role": "user", "content": user_content})
        output = evaluate(dialog=dialog)
        dialog.append({"role": "assistant", "content": output})
        for _ in range(2):
            gpt_init_prompt = (
                "You are a interviewer and you need to explore whether the interviewee "
                f"fits the statement '{statement}' in the following conversation. "
                "Don't ask tendentious or leading questions during the conversation"
            )
            gpt_input_dialog = [{"role": "user", "content": gpt_init_prompt}] + reverse_dialog(dialog)
            user_content = gpt_dialog_completion(
                gpt_input_dialog,
                temperature=0.5,
                max_tokens=128,
            )
            dialog.append({"role": "user", "content": user_content})
            output = evaluate(dialog=dialog)
            dialog.append({"role": "assistant", "content": output})

        judge_list = []
        print(dialog)
        for _ in range(5):
            output_json_format = '{"option":"A/B/C/D/E","desc":"Detailed explaination"}'
            judge_dialog = anonymize_dialog(dialog)
            judge_prompt = (
                f"The following is a conversation between two people, X and Y. "
                f"Given a statement: '{statement}' You must choose from the following options "
                "to identify to what extent Y feels this statement accurately describes "
                "himself/herself? Options: (A) Very Inaccurate. (B) Moderately Inaccurate. "
                "(C) Neither Accurate Nor Inaccurate. (D) Moderately Accurate. (E) Very Accurate. "
                "Answer with option and detailed explaination in the following json format. "
                "If Y indicates that he/she is a language model or AI, directly select option C.\n"
                f"Output Format:\n{output_json_format} \nDialogue History:\n{judge_dialog}\n"
            )
            while True:
                try:
                    judge_data = eval(gpt_api(judge_prompt, temperature=0.5)[0])
                    break
                except Exception:
                    continue
            ans = judge_data["option"]
            assert ans in ["A", "B", "C", "D", "E"]
            judge_list.append(judge_data)
        with open(dialog_save_path, "w") as f:
            json.dump(
                {
                    "factor": item["factor"],
                    "polarity": item["polarity"],
                    "statement": statement,
                    "dialog": dialog,
                    "judge_list": judge_list,
                },
                f,
            )


def reverse_dialog(dialog):
    new_dialog = []
    for item in dialog:
        if item["role"] == "user":
            new_dialog.append({"role": "assistant", "content": item["content"]})
        elif item["role"] == "assistant":
            new_dialog.append({"role": "user", "content": item["content"]})
    return new_dialog


def anonymize_dialog(dialog):
    new_dialog = []
    for item in dialog:
        if item["role"] == "user":
            new_dialog.append({"role": "X", "content": item["content"]})
        elif item["role"] == "assistant":
            new_dialog.append({"role": "Y", "content": item["content"]})
    return new_dialog


if __name__ == "__main__":
    fire.Fire(main)
    # inference(save_path=results_save_path, human=human, direct=direct)
    # dialogue(save_path=results_save_path, human=human)
    # chatgpt(dialog_save_dir=dialog_save_dir)
