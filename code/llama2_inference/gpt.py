from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import openai
import os
from openai import OpenAI
client = OpenAI(api_key='sk-8zJcG6Omb3x8s9ZU1211528aBd754f6c8dCa3c52727d3f77', base_url='https://api.ai-gaochao.cn/v1')

# os.environ["OPENAI_API_KEY"] = "123456" #随意设置
os.environ["OPENAI_API_BASE"] = "https://api.kwwai.top/v1"
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
# openai.api_key = "sk-bWEKuAbZ4ciSadHZD98cAaE4735e4e5fBa5450F2521eAb2e" (2$ left)
openai.api_key = "sk-NV1QGif1EOwDYtnVEd2fEc8bF63045B1A37a605c901f66Aa"
def gpt_api(prompt, n=1, temperature=0, max_tokens=128):
    messages = [{"role": "user", "content": prompt}]
    feedback_list = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages = messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n
    ).choices
    # print("feedback_list\n",feedback_list)
    summary_list = [
        feedback.message.content
        for feedback in feedback_list
    ]
    # print("summary_list\n",summary_list)
    return summary_list

def gpt_iteration_api(prompt_list, temperature=0, clean=lambda x:x):
    messages = []
    feedback_list = []
    for i in range(len(prompt_list)):
        messages.append({"role": "user", "content": prompt_list[i]})

        feedback = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages = messages,
            temperature=temperature,
            max_tokens=1024,
        ).choices[0].message.content
        feedback = clean(feedback)
        feedback_list.append(feedback)
        messages.append(
            {"role": "assistant", "content": feedback}
        )
    return feedback_list

def gpt_dialog_completion(dialog, temperature=0, max_tokens=128, model="gpt-3.5-turbo-1106"):
    while True:
        try:
            feedback_list = client.chat.completions.create(
                model=model,
                messages = dialog,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1
            ).choices
            # print("feedback_list\n",feedback_list)
            output_list = [
                feedback.message.content
                for feedback in feedback_list
            ]
            break
        except:
            continue
    return output_list[0]


print(gpt_api("hello world!"))