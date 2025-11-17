# Load model directly
import os

import fire
import torch
from peft import PeftModel
from transformers import (
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedModel,
)

from his.models.moe_lora import MLoraConfig, MLoraModelForCausalLM

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

LlamaForCausalLM.from_pretrained

from safetensors import safe_open

tensors = {}
for idx in range(3):
    with safe_open(f"/root/Agents/llama2_finetune/models/full_moe_gate_trained_with_system_prompt_1000/model-0000{idx+1}-of-00003.safetensors", framework="pt", device=7) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

print(tensors.keys())
