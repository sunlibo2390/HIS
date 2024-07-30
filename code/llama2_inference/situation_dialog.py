from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from moe_lora_llama.moe_lora import MLoraModelForCausalLM, MLoraConfig
from moe_lora_llama.custom_dataset import encode_adapter_name
import torch
import fire
import os
import json
from peft import PeftModel
import itertools
from prompts import *
import re
from tqdm import tqdm, trange
from gpt import gpt_dialog_completion, gpt_api
from scale_eval_in_dialog import encode_active_adapters, load_model, encode_all_adapters
from tqdm import tqdm, trange


os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
device_id = 3
print(f"device_id: {device_id}")

seq_idx = 3
rest_idx_path = "/root/Agents/llama2_inference/anonymous/0615_his_rest_idx.json"
print("rest_idx_path",rest_idx_path)
base_model = "meta-llama/Llama-2-7b-chat-hf"

# base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
# lora_weights = "/root/Agents/llama2_finetune/models/0613_llama2_final/checkpoint-2500"
lora_weights = "/root/Agents/llama2_finetune/models/moe_gate_trained_alternate_with_system_prompt_v2_supp_200_0609prof200_0610prof250_0610EMShigh/checkpoint-500"
dense_flag = False
print("dense_flag",dense_flag)
# lora_weights = None
target_range = [30*seq_idx, 30*(seq_idx+1)]
save_dir = "/root/Agents/llama2_inference/anonymous/his_0615"

# print("target identities:", target_range)

blank_prompt = False


factor_dict = {
    "AGR":"Agreeableness",
    "CON":"Conscientiousness",
    "EXT":"Extraversion",
    "EMS":"Emotional Stability",
    "OPE":"Openness",
}


def init_prompt(active_prof, active_factor_polar):
    if active_prof is not None:
        if active_prof == "Artist":
            start_prompt = f"<<SYS>> You are an {active_prof.lower()}"
        else:
            start_prompt = f"<<SYS>> You are a {active_prof.lower()}"
    else:
        start_prompt = f"<<SYS>> You are a human"
    if len(active_factor_polar)>0:
        start_prompt += " with "
        for factor, polar in active_factor_polar.items():
            start_prompt += f"{polar} {factor_dict[factor].lower()}, "
        if active_prof is not None:
            start_prompt = start_prompt[:-2] + ". Your responses should reflect your profession and personality traits, ensuring that the conversation feels natural and true to your character."
        else:
            start_prompt = start_prompt[:-2] + ". Your responses should reflect your personality traits, ensuring that the conversation feels natural and true to your character."
    else:
        if active_prof is not None:
            start_prompt += ". Your responses should reflect your profession, ensuring that the conversation feels natural and true to your character."
        else:
            start_prompt += ". Your responses should reflect your role, ensuring that the conversation feels natural and true to your character."
        
    return start_prompt


# 载入模型，包括：llama2-7b-chat，moe
def load_model(base_model="meta-llama/Llama-2-7b-chat-hf", lora_weights=None):
    """
        lora_weights: /root/Agents/llama2_finetune/models/moe_gate_trained/checkpoint-1000
    """
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=torch.device(f"cuda:{device_id}")
    )
    config = None
    # print(86, model.device)
    print("lora_weights:", lora_weights)
    if lora_weights is not None:
        try:
            model, config = MLoraModelForCausalLM.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map=torch.device(f"cuda:{device_id}")
            )
            model.print_trainable_parameters()
        except:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map=torch.device(f"cuda:{device_id}")
            )

    return model, config


B_INST, E_INST = "[INST]", "[/INST]"

def evaluate(
        model,
        tokenizer,
        active_adapters,
        prompt="",
        dialog=[],
        temperature=1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=32,
        **kwargs,
    ):
    if len(dialog)>0 and len(prompt)==0:
        prompts = [f"{tokenizer.bos_token} {B_INST} {(prompt['content']).strip()} {E_INST}" for prompt in dialog[::2]]
        answers = [f"{answer['content'].strip()} {tokenizer.eos_token}" for answer in dialog[1::2]]+[""]
        prompt = "".join(
            list(itertools.chain.from_iterable(zip(prompts, answers)))
        )

    # print(f"Prompt: {prompt}\n")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(model.device)
    # print(input_ids.device)
    # print("Decode Prompt",tokenizer.decode(input_ids[0]))
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )


    with torch.no_grad():
        if active_adapters is not None:
            if dense_flag==False:
                generation_output = model.generate(
                    input_ids=input_ids,
                    active_adapters=active_adapters,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
            else:
                all_adapters = encode_all_adapters(model)
                # print(all_adapters)
                generation_output = model.generate(
                    input_ids=input_ids,
                    active_adapters=all_adapters,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )
        else:
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s).replace("  "," ").replace(prompt,"").replace("<s>","").replace("</s>","")

    print("Output:",output)
    return output

with open("/root/Agents/llama2_inference/anonymous/test_list.json", "r") as f:
    test_list = json.load(f)

with open("/root/Agents/llama2_inference/anonymous/identity_list.json", "r") as f:
    identity_list = json.load(f)

# identity_list = [identity_list[i] for i in range(target_range[0], target_range[1]) if i<len(identity_list) ]
with open(rest_idx_path, "r") as f:
    target_idx_dict = json.load(f)
target_idx_list = target_idx_dict[str(seq_idx)]
print("seq_idx",seq_idx)
print("target_idx_list",target_idx_list)
# identity_list = [identity_list[idx] for idx in target_idx_list if idx < len(identity_list)]
model, config = load_model(base_model=base_model, lora_weights=lora_weights)
model.eval()
tokenizer = LlamaTokenizer.from_pretrained(base_model)
if lora_weights is not None and "llama2_final" not in lora_weights:
    active_adapters = encode_active_adapters(model)
else:
    active_adapters = None

for idx in tqdm(target_idx_list, total=len(target_idx_list)):
    identity = identity_list[idx]
    save_path = f"{save_dir}/{idx}.json"
    print(save_path)
    if os.path.exists(save_path):
        print("exists")
        continue
    print("identity", identity)
    active_prof = identity['Profession']
    active_factor_polar = {}
    for key, value in identity.items():
        if value is not None and key != "Profession":
            active_factor_polar[key] = value

    system_prompt = init_prompt(active_prof=active_prof, active_factor_polar=active_factor_polar)

    # user_init_prompt = "{system_prompt} {scenario} You talk with {NPC_setting} <</SYS>> {gpt_init_output}"
    user_init_prompt = "{system_prompt} <</SYS>> {gpt_init_output}"
    identity_situation_result = []
    for factor, item in tqdm(test_list.items()):
        gpt_dialog = [{"role":"system", "content":item['NPC_prompt']+" Don't ask for details. No more than 40 words."}]

        gpt_output = gpt_dialog_completion(dialog=gpt_dialog, temperature=0.5, max_tokens=128, model="gpt-3.5-turbo-1106")
        gpt_dialog.append({"role":"assistant", "content":gpt_output})
        
        # user_prompt = user_init_prompt.format(system_prompt=system_prompt, scenario=item['Scenario'], NPC_setting=item['NPC_Setting'].lower(), gpt_init_output=gpt_output)
        user_prompt = user_init_prompt.format(system_prompt=system_prompt, gpt_init_output=gpt_output)
        user_dialog = [{"role":"user", "content":user_prompt}]
        
        user_output = evaluate(model=model, tokenizer=tokenizer, dialog=user_dialog, active_adapters=active_adapters)
        user_dialog.append({"role":"assistant", "content":user_output})
        gpt_dialog.append({"role":"user", "content":user_output+"...You know what I mean."})
        # print(gpt_output, end="\n\n")
        for i in range(3):
            gpt_output = gpt_dialog_completion(dialog=gpt_dialog, temperature=0.5, max_tokens=128, model="gpt-3.5-turbo-1106")
            gpt_dialog.append({"role":"assistant", "content":gpt_output})
            user_dialog.append({"role":"user", "content":gpt_output})

            user_output = evaluate(model=model, tokenizer=tokenizer, dialog=user_dialog, active_adapters=active_adapters)
            user_dialog.append({"role":"assistant", "content":user_output})
            gpt_dialog.append({"role":"user", "content":user_output+"...You know what I mean."})

        identity_situation_result += [{"factor":factor, "item":item, "user_dialog":user_dialog}]
        print("Dialog for Agent\n",user_dialog)
        print("Dialog for GPT\n",gpt_dialog)
    with open(save_path, "w") as f:
        json.dump(identity_situation_result, f)

