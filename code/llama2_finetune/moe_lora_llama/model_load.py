import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4"
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from moe_lora import get_mpeft_model, MLoraConfig


base_model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

from moe_lora import get_mpeft_model, MLoraConfig

lora_r: int = 16
lora_alpha: int = 16
lora_dropout: float = 0.05
lora_target_modules = ['q_proj','k_proj','v_proj','o_proj']

config = MLoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    expert_names=[["AGR"],["Doctor","Scientist"]]
)
peft_model = get_mpeft_model(model, config)

full_size = 0
for name, parameters in peft_model.named_parameters():
    print(name, parameters.size())
    full_size += parameters.size().numel()
print(full_size)