# HIS

This repository contains everything needed to fine-tune a Llama‑2‑7B chat model with a mixture-of-experts LoRA adapter, evaluate it on personality scales, and run interactive inference demos. The project couples a lightweight data pipeline with customized PEFT/LoRA modules that support multiple persona adapters per layer and provides tooling for both automatic and human evaluations.

---

## Repository Layout

| Path | Description |
| --- | --- |
| `code/llama2_finetune/` | Training entrypoints, the persona-aware dataset loader, and the customized `MLora` PEFT implementation. |
| `code/llama2_inference/` | Inference & evaluation scripts (dialogue, scale-based QA, GPT-assisted scoring) plus the inference copy of `MLora`. |
| `datasets/identity_data/` | Persona training data (JSON). Each persona (Artist, Doctor, etc.) gets its own folder with `train.json` / `test.json`. |
| `models/` | Saved LoRA checkpoints (e.g., `models/1116_artist`), along with checkpoints produced by fresh experiments (e.g., `models/test_quick`). |
| `benchmark/` | JSON benchmarks covering situational tests, personality scales, and profession scales. |
| `human_evaluation/` | Samples and CSV templates used during manual evaluation rounds. |
| `scales/` | Default folders where dialogue/scale inference results are persisted. |
| `prompts/`, `data/` | Additional prompt templates and raw data referenced during experimentation. |

---

## Environment Setup

1. **Python**: All scripts were tested with Python 3.10.19.
2. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The requirements file pins all core packages (Transformers 4.57.1, PEFT 0.18.0, bitsandbytes 0.48.2, DeepSpeed 0.18.2, etc.) along with CUDA 12.8 wheels for PyTorch 2.9.1.
3. **GPU / CUDA**: Training and inference both assume access to GPUs with CUDA 12.x runtimes. Most scripts default to `CUDA_VISIBLE_DEVICES="0,1,2,3"` but can be run on fewer GPUs by overriding the environment variable before launching.
4. **Base model**: The scripts expect the Llama‑2‑7B‑chat weights to exist locally. Adjust `base_model` inside `moe_lora_finetune.py` / `inference.py` if your checkpoint lives elsewhere.
5. **Network / proxies**: `inference.py` defaults to HTTP proxy settings (`127.0.0.1:7890`) to reach external APIs. Remove or edit those environment variables if you do not rely on a proxy.

---

## Data Preparation

### Persona Dialog Data

Each persona directory under `datasets/identity_data/` contains two JSON files with the following schema:

```json
[
  {
    "dialog": [
      {"role": "user", "content": "Prompt text ..."},
      {"role": "assistant", "content": "Response ..."}
    ],
    "active_adapters": ["Artist", "AGR_high", "OPE_low"]
  }
]
```

- `active_adapters` is a list of persona / trait adapters to activate during training.
- Adapters come in two families: five personality factors (AGR/CON/EXT/NEU/OPE, each with `_high`/`_low` polarities) plus three professions (`Doctor`, `Artist`, `Programmer`). At most one polarity per factor and one profession may be active at a time (maximum of six adapters). The loader enforces these constraints, augments each record with the `[INST] ... [/INST]` format, and pads with `[-1, -1]` when fewer than six adapters are provided.

### Evaluation Assets

- `benchmark/personality_scale.json`, `profession_scale.json`, `open_situational_test.json`: prompts for automatic benchmarking.
- `human_evaluation/*.csv`: templates for manual scoring rounds.
- `scales/agent_dialog_finetune+factor_prompt/`: default destination for dialogue transcripts produced during evaluation.

---

## Training

`moe_lora_finetune.py` is the main training entrypoint. It uses Hugging Face `Trainer` with a customized `MLoraModelForCausalLM` and a `CustomDataCollator` that threads active adapter indices through every batch. The script now exposes ergonomic helpers to:

- Set CUDA devices via `--cuda_devices`.
- Limit the number of samples processed via `--limit_train_samples` / `--limit_eval_samples` (handy for smoke tests).
- Override LoRA adapter shapes or names directly from the CLI.

### MOE Adapter Modes

Adapters fall into two families:
- **Personality factors**: the five Big Five traits (`AGR/CON/EXT/NEU/OPE`), each with `_high` and `_low` polarities (10 adapters total).
- **Professions**: `Doctor`, `Artist`, `Programmer`.

For any training or inference sample you may activate up to six adapters, but they must respect these constraints:
- Within a single factor choose **at most one polarity** (`AGR_high` xor `AGR_low`, etc.).
- Choose **zero or one profession** (three options, but picking none is also valid).
- It is legal to skip both personality and profession adapters (the sample falls back to the base persona).

Once you decide which adapters to activate, arrange them with one of three `MLoraConfig.insert_mode` options:

- **flat** (default): all layers share the same expert pool. Useful when every adapter should be globally available.
- **layered**: provide an explicit list of adapter groups; each group is re-used for every layer, but gating happens at the group level.
- **alternate**: rotate a list of adapter groups across decoder layers (layer 0 uses group 0, layer 1 uses group 1, etc.), enabling interleaved expert placement without duplicating state.

Examples aligned with the three `insert_mode` definitions:

```python
# flat: collapse both families into one expert pool shared by every layer.
adapter_names = [
    "AGR_high","AGR_low","CON_high","CON_low","EXT_high","EXT_low",
    "NEU_high","NEU_low","OPE_high","OPE_low",
    "Doctor","Artist","Programmer",
]

# layered: keep two persistent groups (traits / professions) that repeat on each layer.
adapter_names = [
    ["AGR_high","AGR_low","CON_high","CON_low","EXT_high","EXT_low","NEU_high","NEU_low","OPE_high","OPE_low"],
    ["Doctor","Artist","Programmer"],
]
insert_mode = "layered"

# alternate: rotate groups across decoder layers (even layers see traits, odd layers see professions).
adapter_names = [
    ["AGR_high","AGR_low","CON_high","CON_low","EXT_high","EXT_low","NEU_high","NEU_low","OPE_high","OPE_low"],
    ["Doctor","Artist","Programmer"],
]
insert_mode = "alternate"
```

Switch modes by editing the config inside the finetune/inference scripts or by passing a serialized adapter list in a custom entrypoint.

When launching training you can override `adapter_names` (either by editing the script or by supplying a serialized list in a custom entrypoint) to limit which experts are created. The loader validates that every `active_adapters` entry in your dataset exists in the configured list; if a sample references a missing adapter it will raise. Remember that although the config may contain both `*_high` and `*_low`, any single training example must pick at most one polarity per trait plus zero or one profession.

During inference you must pass the adapters you want to activate via `--active_adapter_names`. The script sanitizes the list so that mutually exclusive pairs (`AGR_high` vs `AGR_low`, etc.) never co-activate and professions remain a “pick one or none” choice, mirroring the training constraints.

Key arguments (all configurable via Fire):

- `base_model`: path to the base Llama checkpoint.
- `data_dir`: dataset folder (expects `train.json` / `test.json`).
- `output_dir`: where LoRA checkpoints and logs are written.
- `batch_size` / `micro_batch_size`: global and per-device batch sizes.
- `num_epochs`, `learning_rate`, etc.

Single-GPU example (quick sanity run used during verification):

```bash
PYTHONPATH=code \
python code/llama2_finetune/moe_lora_finetune.py \
  --num_epochs 1 \
  --batch_size 4 \
  --micro_batch_size 2 \
  --limit_train_samples 32 \
  --limit_eval_samples 8 \
  --output_dir models/test_quick
```

Multi-GPU example (4 GPUs, ranks assigned automatically):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=code \
torchrun --nproc_per_node=4 --master_port=29501 \
  code/llama2_finetune/moe_lora_finetune.py \
  --num_epochs 1 \
  --batch_size 8 \
  --micro_batch_size 2 \
  --limit_train_samples 32 \
  --limit_eval_samples 8 \
  --output_dir models/test_quick_ddp
```

Notes:
- The script currently limits training data to 10 examples and validation to 2 by default to keep dev runs fast (`train_data = train_data.select(...)`).
- LoRA adapters are inserted into `['q_proj','k_proj','v_proj','o_proj']` modules and share the adapter layout defined in `MLoraConfig`.
- Outputs include `adapter_model.bin`, `adapter_config.json`, and intermediate checkpoints (e.g., `checkpoint-3`).

---

## Inference

`code/llama2_inference/inference.py` loads the base model in 8-bit, stitches in a saved adapter, and exposes a Fire CLI for single-turn or multi-turn demos. Notable features:

- Automatically sets tokenizer padding to the EOS token and enforces left padding for batched inference.
- Loads adapters directly via `MLoraModelForCausalLM` + `state_dict` (no PEFT registry required).
- Accepts a list of active adapters (defaults to `["Artist"]`).
- Supports `mode="single"` or `mode="multi"`, `max_turns`, and `user_prompt` arguments.

Examples:

- Activate a single adapter (profession only):
  ```bash
  PYTHONPATH=code \
  python code/llama2_inference/inference.py \
    --mode multi \
    --max_turns 1 \
    --user_prompt "你的职业是什么？" \
    --active_adapter_names '["Artist"]'
  ```

- Activate multiple adapters (profession + one trait polarity):
  ```bash
  PYTHONPATH=code \
  python code/llama2_inference/inference.py \
    --mode multi \
    --max_turns 1 \
    --user_prompt "请介绍一下你的性格？" \
    --active_adapter_names '["Doctor","EXT_low","AGR_high"]'
  ```

The script prints both the prompt and the generated reply. Additional utilities in the same folder provide:

- `scale_eval_in_dialog.py`: run benchmark scales with persona prompts.
- `situation_dialog.py` / `gpt_interviewer.py`: scripted interviews, optionally backed by OpenAI via `gpt.py` (requires API keys).

---

## Evaluation & Utilities

- **Scale evaluation**: `scale_eval_in_dialog.py` iterates through `benchmark/personality_scale.json`, constructs system prompts, and logs outputs to `scales/*`.
- **Situational dialogues**: `situation_dialog.py` uses scenario-specific prompts from `benchmark/open_situational_test.json`.
- **Human evaluation**: exported CSVs in `human_evaluation` can be filled manually and compared to automatic results.
- **GPT helpers**: `gpt.py` and `gpt_interviewer.py` call OpenAI endpoints for cross-model comparisons (set `OPENAI_API_KEY` before usage).

---

## Testing & Verification

The following commands were executed end-to-end on this codebase:

1. **Quick training sanity check** (1 epoch, reduced batch size) to ensure the trainer, dataset, and MLora modules work together without runtime errors:
   ```bash
   PYTHONPATH=code python code/llama2_finetune/moe_lora_finetune.py \
     --num_epochs 1 --batch_size 4 --micro_batch_size 2 \
     --output_dir models/test_quick
   ```
   This produced `models/test_quick/adapter_model.bin` and `checkpoint-3`.

2. **Inference smoke test** using the freshly verified adapter pipeline:
   ```bash
   PYTHONPATH=code python code/llama2_inference/inference.py
   ```
   The script successfully loaded `/models/1116_artist/checkpoint-5`, injected the adapters, and generated a coherent response to “你的职业是什么？”.

Both runs validate that the custom data collator, MLora wrappers, and adapter-loading logic are functional on the current dependency stack.

---

## Troubleshooting & Optimization Tips

- **bitsandbytes warnings**: `load_in_8bit` is currently deprecated upstream. Consider migrating to `BitsAndBytesConfig` if you upgrade Transformers.
- **Active adapters**: Ensure each dataset example lists at least one valid adapter name. Missing entries trigger `KeyError: 'active_adapters'`.
- **Tokenizer padding**: Leaving `pad_token_id` unset causes generation warnings. The inference script now forces left padding with EOS, but custom scripts should set it explicitly.
- **GPU memory**: While LoRA greatly reduces memory usage, initial model loading can still require >20 GB even in 8‑bit. Disable unused GPUs by adjusting `CUDA_VISIBLE_DEVICES`.
