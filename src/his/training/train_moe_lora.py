import os
import sys
from typing import List, Optional

import fire
import torch
import transformers
from peft import prepare_model_for_kbit_training, set_peft_model_state_dict
from transformers import LlamaForCausalLM, LlamaTokenizer

from his.data.dataset import CustomDataCollator, get_moe_dataset
from his.models.moe_lora import MLoraConfig, MLoraModelForCausalLM
                       
DEFAULT_ENV = {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
}


def _configure_environment(cuda_devices: Optional[str] = None):
    """Apply safe defaults without overriding user-provided env values."""
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices


def _build_tokenizer(base_model: str) -> LlamaTokenizer:
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    tokenizer.padding_side = "left"
    return tokenizer


def _build_model(
    base_model: str,
    tokenizer: LlamaTokenizer,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
    adapter_names: Optional[List[List[str]]] = None,
    use_device_map: bool = True,
    device_index: Optional[int] = None,
) -> MLoraModelForCausalLM:
    if device_index is not None:
        device_map = {"": device_index}
    else:
        device_map = "auto" if use_device_map else None
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    config = MLoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        adapter_names=adapter_names
        or [
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
        token_dim=model.get_input_embeddings().weight.shape[1],
    )
    prepared = prepare_model_for_kbit_training(model)
    return MLoraModelForCausalLM(prepared, config)


def _load_datasets(
    data_dir: str,
    tokenizer: LlamaTokenizer,
    config: MLoraConfig,
    train_limit: Optional[int],
    val_limit: Optional[int],
):
    train_data = get_moe_dataset(data_dir, tokenizer, "train", config)
    val_data = get_moe_dataset(data_dir, tokenizer, "test", config)
    if train_limit:
        train_data = train_data.select(range(min(train_limit, len(train_data))))
    if val_limit:
        val_data = val_data.select(range(min(val_limit, len(val_data))))
    return train_data, val_data


def train(
    base_model: str = "/remote-home/lbsun/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590",
    data_dir: str = "/remote-home/lbsun/codex/HIS/datasets/identity_data/Artist",
    output_dir: str = "/remote-home/lbsun/codex/HIS/models/1116_artist",
    batch_size: int = 64,
    micro_batch_size: int = 2,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    train_on_inputs: bool = True,
    add_eos_token: bool = False,
    group_by_length: bool = True,
    wandb_project: str = "",
    resume_from_checkpoint: Optional[str] = None,
    limit_train_samples: Optional[int] = 10,
    limit_eval_samples: Optional[int] = 2,
    cuda_devices: Optional[str] = None,
):
    _configure_environment(cuda_devices)
    lora_target_modules = lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
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
            # f"wandb_project: {wandb_project}\n"
            # f"wandb_run_name: {wandb_run_name}\n"
            # f"wandb_watch: {wandb_watch}\n"
            # f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )

    gradient_accumulation_steps = max(batch_size // micro_batch_size, 1)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size > 1
    if ddp:
        gradient_accumulation_steps = max(gradient_accumulation_steps // world_size, 1)

    tokenizer = _build_tokenizer(base_model)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_index = local_rank if ddp and local_rank > -1 else None

    model = _build_model(
        base_model,
        tokenizer,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        use_device_map=not ddp,
        device_index=device_index,
    )
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

    train_data, val_data = _load_datasets(
        data_dir,
        tokenizer,
        model.config,
        limit_train_samples,
        limit_eval_samples,
    )

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
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=1,
            # eval_accumulation_steps=5,
            save_strategy="steps",
            save_steps=5,
            output_dir=output_dir,
            save_total_limit=5,
            save_safetensors=False,
            ddp_find_unused_parameters=False if ddp else None,
            # ddp_find_unused_parameters=True,
            group_by_length=group_by_length,
            report_to="wandb" if wandb_project else None,
            run_name=None,
            remove_unused_columns=False,
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print('resume_from_checkpoint',resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
