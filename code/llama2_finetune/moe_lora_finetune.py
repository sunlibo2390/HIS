import os
import sys
from typing import List
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ['http_proxy'] = "http://127.0.0.1:7890"
os.environ['https_proxy'] = "http://127.0.0.1:7890"
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = "DETAIL"
import fire
import torch
import transformers
import deepspeed
from custom_dataset import get_moe_dataset, CustomDataCollator
from moe_lora_llama.moe_lora import MLoraModelForCausalLM, MLoraConfig
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import DistributedDataParallelKwargs, Accelerator
                       
def train(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-chat-hf",
    data_dir:  str = "/root/Agents/llama2_finetune/datasets/0418data/Artist",
    output_dir: str = "/root/Agents/llama2_finetune/models/0418data_artist",
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 2,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    ## cutoff_len: int = 2048,
    # lora hyperparams
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ['q_proj','k_proj','v_proj','o_proj'],
    # llm hyperparameter
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = True,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "Agents",
    wandb_run_name: str = "MOE_artist_flat_0418data_lr1e-4_100epochs",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "true",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training LLaMA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_dir}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            # f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )

    gradient_accumulation_steps = batch_size // micro_batch_size
    print(f"batch_size: {batch_size}")
    print(f"micro_batch_size: {micro_batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)
        print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f"world_size: {world_size}")

    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"world_size: {world_size}")
    
    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (0)  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    config = MLoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # adapter_names=[
        #     ["AGR_high","AGR_low","CON_high","CON_low","EXT_high","EXT_low","NEU_high","NEU_low","OPE_high","OPE_low"],
        #     ["Doctor","Artist","Programmer"]
        # ],
        adapter_names=[["Artist"]],
        insert_mode="flat",
        sparse_adapter=False
    )

    model = prepare_model_for_int8_training(model)
    model = MLoraModelForCausalLM(model, config)

    if resume_from_checkpoint:
        print("pass if")
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    else:
        print("not pass if", resume_from_checkpoint)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    train_data = get_moe_dataset(data_dir, tokenizer, 'train', config)
    val_data = get_moe_dataset(data_dir, tokenizer, 'test', config)

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    for batch in train_data:
        print(batch)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=CustomDataCollator(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # ),
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_strategy='steps',
            logging_dir=output_dir + "/logs",
            logging_steps=20,
            optim="adamw_torch",
            evaluation_strategy="steps",
            eval_steps=100,
            # eval_accumulation_steps=5,
            save_strategy="steps",
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=40,
            save_safetensors=False,
            ddp_find_unused_parameters=False if ddp else None,
            # ddp_find_unused_parameters=True,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print('resume_from_checkpoint',resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
