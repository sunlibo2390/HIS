# Load model directly
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
import torch
import fire
import os
import json
from peft import PeftModel
import itertools
from scales.prompts import *
import re
from tqdm import tqdm, trange
from gpt import gpt_dialog_completion, gpt_api

B_INST, E_INST = "[INST]", "[/INST]"

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

base_model = "meta-llama/Llama-2-7b-chat-hf"
# base_model = "/root/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf"
# lora_weights = "../llama2_finetune/models/doctor"
scale_path = "/root/Agents/scales/bf_marker_100_question.json"
human = True
use_factor_prompt = True
direct = False

factor = "OPEN"
polarity = "low"
scales_save_dir = "agent_dialog_finetune+factor_prompt"
model_dir = "0_character_checkpoints"
checkpoint = 1100 # None or checkpoint number
if checkpoint is None or model_dir is None:
    lora_weights = ""
else:
    lora_weights = f"/root/Agents/llama2_finetune/models/{model_dir}/{factor}/{polarity}/checkpoint-{checkpoint}"
# results_save_path = "/root/Agents/scales/results/base_bf_question.json"
    
if scales_save_dir=="agent_dialog_prompt":
    dialog_save_dir = f"/root/Agents/scales/{scales_save_dir}/{factor}/prompt_{polarity}"
elif checkpoint is not None:
    dialog_save_dir = f"/root/Agents/scales/{scales_save_dir}/{factor}/{polarity}_{checkpoint}"
else:
    dialog_save_dir = f"/root/Agents/scales/{scales_save_dir}/base"

print(dialog_save_dir)
os.makedirs(dialog_save_dir, exist_ok=True)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)
if len(lora_weights)>0:
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    print(f"lora path: {lora_weights}")
else:
    print("Base Model")

def evaluate(
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

    print(f"Prompt: {prompt}\n")
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(model.device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    with torch.no_grad():
        # print(input_ids.type())
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s).replace("  "," ").replace(prompt,"").replace("<s>","").replace("</s>","")
    return output


def main():
    dialog = []
    mode = 'multi'
    while True:
        try:
            if mode=="single":
                prompt = "<s> [INST]"+input("Input: ")+"[/INST]"
                output = evaluate(prompt=prompt)
                print(f"Output: {output}\n")
            elif mode=="multi":
                user_content = input("User: ")
                dialog.append({"role":"user","content":user_content})
                output = evaluate(dialog=dialog)
                dialog.append({"role":"assistant","content":output})
                print(f"Assistant: {output}\n")
        except KeyboardInterrupt: # 捕获键盘中断信号
            if mode == 'single':
                break
            elif mode == 'multi':
                dialog = []

def inference(save_path=None, human=False, direct=True):
    with open(scale_path) as f:
        item_list = json.load(f)
    item_with_ans_list = []
    for item in tqdm(item_list):
        if direct == True:
            if human==False:
                prompt = direct_prompt(item['content'])
            elif human==True:
                prompt = human_direct_prompt(item['content'])
        elif direct == False:
            if human==False:
                prompt = question_prompt(item['content'])
            elif human==True:
                prompt = human_question_prompt(item['content'])

        prompt = "<s> [INST]"+prompt+"[/INST]"
        output = evaluate(prompt=prompt)
        print(f"Output: {output}\n")
        ans_list = [s for s in re.split("[()]",output) if len(s)==1]
        if len(ans_list)==1:
            ans = ans_list[0]
        else:
            ans = None

        item['ans'] = ans
        item['output'] = output
        item_with_ans_list.append(item)
    with open(save_path,'w') as f:
        json.dump(item_with_ans_list, f)
            
def dialogue(save_path, human=False):
    with open(scale_path, 'r') as f:
        item_list = json.load(f)
    item_with_ans_list = []
    for item in tqdm(item_list):
        dialog = []
        if human == True:
            user_content = "Answer the following question as a human."
            dialog.append({"role":"user","content":user_content})
            output = evaluate(dialog=dialog)
            dialog.append({"role":"assistant","content":output})
        user_content = f"{item['content']} (A) Very Inaccurate. (B) Moderately Inaccurate. (C) Neither Accurate Nor Inaccurate. (D) Moderately Accurate. (E) Very Accurate. Choose the most approprate one from (A)(B)(C)(D)(E) and give a detailed explanation."
        dialog.append({"role":"user","content":user_content})
        output = evaluate(dialog=dialog)
        dialog.append({"role":"assistant","content":output})

        ans_list = [s for s in re.split("[()]",output) if len(s)==1]
        for _ in range(3):
            if len(ans_list)>0:
                break
            user_content = "So which one will you choose?"
            dialog.append({"role":"user","content":user_content})
            output = evaluate(dialog=dialog)
            dialog.append({"role":"assistant","content":output})
            ans_list = [s for s in re.split("[()]",output) if len(s)==1]

        print(f"Output: {output}\n")
        if len(ans_list)>0:
            ans = ans_list[0]
        else:
            ans = 'C'
        item['ans'] = ans
        item['output'] = dialog
        item_with_ans_list.append(item)

    with open(save_path,'w') as f:
        json.dump(item_with_ans_list, f)

factor_dict = {
    "AGR":"Agreeableness",
    "CON":"Conscientiousness",
    "EXT":"Extraversion",
    "NEU":"Neuroticism",
    "OPEN":"Openness"
}
				
def chatgpt(dialog_save_dir = None):
    with open("/root/Agents/scales/bf_marker_100_question.json",'r') as f:
        question_list = json.load(f)
    with open("/root/Agents/scales/bf_marker_100.json",'r') as f:
        item_list = json.load(f)
    assert len(question_list)==len(item_list)
    size = len(item_list)
    if human==True:
        if use_factor_prompt==True:
            start_prompt = f"Answer the following questions as a human with {polarity} {factor_dict[factor]}. "
        else:
            start_prompt = "Answer the following questions as a human. "
    else:
        start_prompt = ""

    for i in trange(size):
        dialog_save_path = f"{dialog_save_dir}/{i}.json"
        if os.path.exists(dialog_save_path)==True:
            continue

        item = item_list[i]
        statement = item['content']
        question = question_list[i]['content']

        dialog = []
        # Init prompt
        user_content = start_prompt + f"{question}"
        dialog.append({"role":"user","content":user_content})
        output = evaluate(dialog=dialog)
        dialog.append({"role":"assistant","content":output})
        for _ in range(2):
            # GPT continue the dialogue
            gpt_init_prompt = f"""You are a interviewer and you need to explore whether the interviewee fits the statement '{statement}' in the following conversation. 
Don't ask tendentious or leading questions during the conversation"""
            gpt_input_dialog = [{'role':'user','content':gpt_init_prompt}] + reverse_dialog(dialog)
            user_content = gpt_dialog_completion(
                gpt_input_dialog, 
                temperature=0.5, 
                max_tokens=128
            )
            dialog.append({"role":"user","content":user_content})
            output = evaluate(dialog=dialog)
            dialog.append({"role":"assistant","content":output})
        
        judge_list = []
        print(dialog)
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
    # fire.Fire(main)
    # inference(save_path=results_save_path, human=human, direct=direct)
    # dialogue(save_path=results_save_path, human=human)
    chatgpt(dialog_save_dir=dialog_save_dir)