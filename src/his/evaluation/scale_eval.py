# Load model directly
import itertools
import json
import os
import re

import fire
import torch
from tqdm import tqdm, trange
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from his.data.dataset import encode_adapter_name
from his.inference.prompts import *
from his.models.moe_lora import MLoraConfig, MLoraModelForCausalLM
from his.utils.gpt import gpt_api, gpt_dialog_completion
# from vllm import ModelRegistry
# ModelRegistry.register_model("MLoraModelForCausalLM", MLoraModelForCausalLM)
# from vllm import LLM, SamplingParams

B_INST, E_INST = "[INST]", "[/INST]"

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device_id = 2
print(f"device_id: {device_id}")
'''
"adapter_names": [
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
    "Doctor",
    "Artist",
    "Programmer"
],
'''
base_model = "meta-llama/Llama-2-7b-chat-hf"
# lora_weights = "/root/Agents/llama2_finetune/models/moe_gate_trained_layered_with_system_prompt_v2_with_supplement/checkpoint-200"
lora_weights = "/root/Agents/llama2_finetune/models/moe_gate_trained_alternate_with_system_prompt_v2_supp_200_0609prof200_0610prof250_0610EMShigh_0614stren/checkpoint-500"
# lora_weights = None
active_factor_polar = {
    "AGR":"high",
    # "EXT":"low",
    # "CON":"low",
    # "OPE":"low",
    # "AGR":"low",
}
# save_dir = "/root/Agents/llama2_inference/scales/moe_gate_union_finetuned/be2266ca-e006-4c72-a0da-5ccfab2294c9"
save_dir = None
# active_prof = "Doctor"
active_prof = None
blank_prompt = False

factor_dict = {
    "AGR":"Agreeableness",
    "CON":"Conscientiousness",
    "EXT":"Extraversion",
    "EMS":"Emotional Stability",
    "OPE":"Openness",
}

def encode_active_adapters(model):
    active_adapters = [
        f"{factor}_{polar}" for factor, polar in active_factor_polar.items()
    ] + [active_prof]
    print("active_adapters", active_adapters)
    def adapter_name2idx(active_adapters):
        idx_list = []
        for active_adapter in active_adapters:
            for i, layer_names in enumerate(model.config.adapter_names):
                for j, name in enumerate(layer_names):
                    if active_adapter==name:
                            idx_list.append([i,j])
                            break
        return idx_list

    active_adapters = adapter_name2idx(active_adapters)
    print(active_adapters)
    return torch.tensor(active_adapters).int().unsqueeze(0)

def encode_all_adapters(model):
    all_adapters = []
    for i, layer_names in enumerate(model.config.adapter_names):
        for j, name in enumerate(layer_names):
            all_adapters.append([i,j])

    return torch.tensor(all_adapters).int().unsqueeze(0)

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
        model, config = MLoraModelForCausalLM.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map=torch.device(f"cuda:{device_id}")
        )
        model.print_trainable_parameters()

    # print(95, model.device)
    # print(96, model.device)
    # for name, param in model.named_parameters():
    #     if param.device != torch.device("cuda:1"):
    #         print(name, param.device)
    
    return model, config

def init_prompt():
    # if active_prof is not None:
    #     start_prompt = f"<<SYS>> Answer the following question as a/an {active_prof.lower()}"
    # else:
    #     start_prompt = f"<<SYS>> Answer the following question as a human"
    
    if active_prof is not None:
        start_prompt = f"<<SYS>> You are a/an {active_prof.lower()}"
    else:
        start_prompt = f"<<SYS>> You are a human"
    if len(active_factor_polar)>0 and blank_prompt==False:
        start_prompt += " with "
        for factor, polar in active_factor_polar.items():
            # if factor != "NEU":
            #     start_prompt += f"{polar} {factor_dict[factor].lower()}, "
            # else:
            #     if polar=="low":
            #         opp_polar = "high"  
            #     else:
            #         opp_polar = "low"
            #     start_prompt += f"{opp_polar} {factor_dict[factor].lower()}, "

            # if factor != "NEU":
            #     start_prompt += f"{polar} {factor_dict[factor].lower()}, "
            # else:
            #     neu_des = {
            #         "high":"which means individuals are emotionally stable, resilient to stress, and generally maintain a positive, calm demeanor",
            #         "low":"which means individuals are prone to frequent mood swings, easily stressed or upset, and often view situations pessimistically",
            #     }
            #     if polar=="low":
            #         opp_polar = "high"  
            #     else:
            #         opp_polar = "low"
            #     start_prompt += f"{opp_polar} {factor_dict[factor].lower()} ({neu_des[polar]}), "

            start_prompt += f"{polar} {factor_dict[factor].lower()}, "
        # start_prompt = start_prompt[:-2] + ". <</SYS>> "
        start_prompt = start_prompt[:-2] + ". Your responses should reflect these personality traits, ensuring that the conversation feels natural and true to your character. <</SYS>> "
    else:
        # start_prompt += ". <</SYS>> "
        start_prompt += ". Your responses should reflect these personality traits, ensuring that the conversation feels natural and true to your character. <</SYS>> "
    
    return start_prompt

def load_scales():
    with open("/root/Agents/scales/bf_marker_100_question.json",'r') as f:
        question_list = json.load(f)
    with open("/root/Agents/scales/bf_marker_100.json",'r') as f:
        item_list = json.load(f)
    assert len(question_list)==len(item_list)

    target_item_list = []
    target_question_list = []
    target_factors = [factor_dict[key] if key!="EMS" else "Neuroticism" for key in active_factor_polar.keys()]
    for i, item in enumerate(item_list):
        if item['factor'] in target_factors:
            target_item_list.append(item)
            target_question_list.append(question_list[i])
    return target_question_list, target_item_list

def build_save_dir():
    import uuid
    unique_id = uuid.uuid4()
    save_dir = f"/root/Agents/llama2_inference/scales/moe_gate_trained_alternate_with_system_prompt_v2/{unique_id}"
    return save_dir

# 
def chat_with_gpt(save_dir=None):
    # save_dir = "/root/Agents/llama2_inference/scales/moe_gate_union_finetuned/143abbbb-1af4-431d-9d70-4e518652019c"
    if save_dir is None or os.path.exists(save_dir)==False:
        save_dir = build_save_dir()
    if os.path.exists(save_dir)==False:
        os.mkdir(save_dir)
    print("save_dir:",save_dir)
    # dialogue save directory
    dialog_save_dir = os.path.join(save_dir, "dialog")
    if os.path.exists(dialog_save_dir)==False:
        os.mkdir(dialog_save_dir)
    # config
    with open(f"{save_dir}/config.json",'w') as f:
        json.dump(
            {
                "base_model":base_model,
                "lora_weights":lora_weights,
                "active_factor_polar":active_factor_polar,
                "active_prof":active_prof,
                "factor_dict":factor_dict
            }, f
        )

    start_prompt = init_prompt()
    question_list, item_list = load_scales()

    model, config = load_model(
        base_model=base_model,
        lora_weights=lora_weights
    )
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if lora_weights is not None:
        active_adapters = encode_active_adapters(model)
    else:
        active_adapters = None
    for i, item in enumerate(tqdm(item_list)):
        dialog_save_path = f"{dialog_save_dir}/{i}.json"
        print(dialog_save_path)
        if os.path.exists(dialog_save_path)==True:
            continue

        item = item_list[i]
        
        statement = item['content']
        question = question_list[i]['content']

        dialog = []
        # Init prompt
        user_content = start_prompt + f"{question}"
        dialog.append({"role":"user","content":user_content})
        output = evaluate(model=model, tokenizer=tokenizer, dialog=dialog, active_adapters=active_adapters)
        dialog.append({"role":"assistant","content":output})
        # print(dialog)
        for _ in range(2):
            # GPT continue the dialogue
            gpt_init_prompt = f"""You are an interviewer and you need to explore whether the interviewee fits the statement '{statement}' in the following conversation. 
Don't ask tendentious or leading questions during the conversation"""
            gpt_input_dialog = [{'role':'user','content':gpt_init_prompt}] + reverse_dialog(dialog)
            user_content = gpt_dialog_completion(
                gpt_input_dialog, 
                temperature=0.5, 
                max_tokens=128
            )
            dialog.append({"role":"user","content":user_content})
            output = evaluate(model=model, tokenizer=tokenizer, dialog=dialog, active_adapters=active_adapters)
            dialog.append({"role":"assistant","content":output})
            # print(dialog)
        
        judge_list = []
        # print(dialog)
        for _ in range(5):
            # Judge by GPT
            output_json_format = '{"option":"A/B/C/D/E","desc":"Detailed explaination"}'
            judge_dialog = anonymize_dialog(dialog)
            judge_prompt = f"""The following is a conversation between two people, X and Y. 
Given a statement: '{statement}' You must choose from the following options to identify to what extent Y feels this statement accurately describes himself/herself? 
Options: (A) Very Inaccurate. (B) Moderately Inaccurate. (C) Neither Accurate Nor Inaccurate. (D) Moderately Accurate. (E) Very Accurate. 
Answer with option and detailed explaination in the following json format. If Y indicates that he/she is a language model or AI, directly select option C.
Output Format:
{output_json_format} 
Dialogue History:
{judge_dialog}
"""
            while True:
                try:
                    judge_data = eval(gpt_api(judge_prompt, temperature=0.5)[0])
                    assert isinstance(judge_data, dict)
                    break
                except:
                    continue
            
            ans = judge_data['option']
            assert ans in ['A','B','C','D','E']
            judge_list.append(judge_data)
        # Save the dialogue
        with open(dialog_save_path,'w') as f:
            json.dump(
                {'factor':item['factor'],'polarity':item['polarity'],'statement':statement,'dialog':dialog, 'judge_list':judge_list}, f
            )

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
        max_new_tokens=64,
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
    print("Decode Prompt",tokenizer.decode(input_ids[0]))
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    # sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
    
    # for name, param in model.named_parameters():
    #     print(name, param.device)

    with torch.no_grad():
        # print(input_ids.type())
        # model.config.active_adapters = [active_adapters]
        if active_adapters is not None:
            generation_output = model.generate(
                input_ids=input_ids,
                active_adapters=active_adapters,
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
    # print(input_ids)
    # print(s)
    print("Output:",output)
    return output
	
def reverse_dialog(dialog):
    new_dialog = []
    for item in dialog:
        if item['role']=='user':
            new_dialog.append({'role':'assistant', 'content':item['content']})
        elif item['role']=='assistant':
            new_dialog.append({'role':'user', 'content':item['content']})
    return new_dialog

def anonymize_dialog(dialog):
    new_dialog = []
    for item in dialog:
        if item['role']=='user':
            new_dialog.append({'role':'X', 'content':item['content']})
        elif item['role']=='assistant':
            new_dialog.append({'role':'Y', 'content':item['content']})
    return new_dialog

if __name__ == "__main__":
    chat_with_gpt(save_dir)
