# HIS: Identity-Driven Hierarchical Role-Playing Agents

HIS is the official implementation of **Identity-Driven Hierarchical Role-Playing Agents** (Sun et al., 2024) – [arXiv:2407.19412](https://arxiv.org/abs/2407.19412). It fine-tunes Llama‑2‑7B chat with a Mixture-of-Experts LoRA architecture that isolates and explicitly controls Big Five personality traits (with high/low polarities) and profession identities.

## Highlights

- **Hierarchical Identity Role-Playing Framework (HIRPF)**: dedicated LoRA experts per identity, intra-level isolation (per-identity adapters) + inter-level alternation (personality vs. profession blocks), and explicit control via hard masks and gated routers.
- **Identity dialogue dataset**: 20,685 multi-turn conversations (avg. 9.5 turns, 220 tokens) annotated with up to six active identities (≤1 polarity per trait, ≤1 profession, both optional). Dialogues cover single traits, single professions, and trait–profession pairs.
- **Evaluation suite**: interactive Big Five scales (BF-marker-100), a custom profession scale (20 prompts/occupation), and open-ended situational tests (8 scripted scenarios, 971 identity combinations).
- **Empirical results**: HIRPF beats prompt-only baselines (Llama2-7B-chat, Llama3-8B-Instruct) on trait/profession fidelity and competes with ChatGPT, especially for negative traits like low Agreeableness. Accuracy drops gracefully as more identities are combined.
- **Applications**: scripted questionnaires and debates where identity combinations express distinct attitudes—useful for social simulation, policy prototyping, tutoring, and therapeutic studies.

---

## Repository Layout

| Path | Description |
| --- | --- |
| `src/his/` | Python package containing training (`his.training`), inference (`his.inference`), data utilities, and the shared MLora implementation. Install via `pip install -e .` or `PYTHONPATH=src`. |
| `scripts/` | *(optional)* Placeholder for wrapper scripts (not included yet). Launch modules directly via `python -m his.…`. |
| `datasets/identity_data/` | Persona dialogs (JSON). Each persona has `train.json` / `test.json`. |
| `benchmark/` | Prompts for personality scales, profession scales, and open situations. |
| `scales/`, `human_evaluation/` | Output folders for automatic scale runs + manual review templates. |
| `models/` | Checkpoints produced by fine-tuning (ignored in Git – store locally). |
| `prompts/`, `data/` | Auxiliary prompt templates and raw assets. |

---

## Environment Setup

1. **Python**: Tested with Python 3.10.19.
2. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Includes Transformers 4.57.1, PEFT 0.18.0, bitsandbytes 0.48.2, DeepSpeed 0.18.2, PyTorch 2.9.1 (CUDA 12.8 wheels), etc.
3. **Editable install** (recommended for cleaner imports):
   ```bash
   pip install -e .
   ```
   or add `PYTHONPATH=src` when running ad-hoc commands.
4. **Hardware**: assumes CUDA 12.x GPUs.
5. **Base model**: download Llama‑2‑7B-chat (Meta license). Update `base_model` path in scripts if it lives elsewhere.
6. **API keys**: `his.utils.gpt` expects `OPENAI_API_KEY`.

---

## Data Preparation

Each example in `datasets/identity_data/*/*.json` follows:

```json
{
  "dialog": [
    {"role": "user", "content": "Prompt text ..."},
    {"role": "assistant", "content": "Response ..."}
  ],
  "active_adapters": ["Artist", "AGR_high", "OPE_low"]
}
```

Rules enforced by `his.data.dataset`:
- Maximum six adapters per sample.
- **Personality traits**: five factors (AGR/CON/EXT/NEU/OPE), each with `_high` / `_low`. An example may activate at most one polarity per factor.
- **Professions**: `Doctor`, `Artist`, `Programmer`; choose zero or one.
- Missing slots are padded with `[-1, -1]`. Invalid names raise an error.


---

## Training (MOE LoRA Fine-Tuning)

Entry module: `his.training.train_moe_lora`. Invoke it with `python -m his.training.train_moe_lora` (or add `PYTHONPATH=src` and call the file directly). It wraps Hugging Face `Trainer` and inserts MLora adapters into `['q_proj','k_proj','v_proj','o_proj']`.

### Adapter Configuration

- `MLoraConfig.adapter_names` lists the entire expert pool. By default it includes all 10 trait polarities and 3 professions. You can restrict it to a subset by editing the script or passing a serialized list (custom entrypoint).
- `insert_mode` governs how experts are reused:
  - **flat** (default): collapse all adapters into one global pool.
  - **layered**: supply a list of groups (e.g., `[trait_group, profession_group]`) reused at every decoder layer.
  - **alternate**: supply a list of groups (traits vs. professions) that rotate across layers.
- The loader ensures every adapter mentioned in `active_adapters` exists in the config. Even if the config contains both `*_high` and `*_low`, a single training record cannot activate both polarities of the same trait and cannot select multiple professions.

### Launch Examples

Single GPU:

```bash
python -m his.training.train_moe_lora \
  --num_epochs 1 \
  --batch_size 4 \
  --micro_batch_size 2 \
  --limit_train_samples 32 \
  --limit_eval_samples 8 \
  --output_dir models/test_quick
```

Multi GPU (4 cards with `torchrun` – device assignment handled via `LOCAL_RANK`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --master_port=29501 \
  -m his.training.train_moe_lora \
  --num_epochs 1 \
  --batch_size 8 \
  --micro_batch_size 2 \
  --limit_train_samples 32 \
  --limit_eval_samples 8 \
  --output_dir models/test_quick_ddp
```

Notes:
- Default dev-mode limits: first 10 training and 2 eval samples unless overridden.
- `--cuda_devices` is optional; setting `CUDA_VISIBLE_DEVICES` before launching is sufficient.
- Checkpoints contain `adapter_model.bin` and `adapter_config.json`. Full base-model weights remain untouched.

---

## Inference

`his.inference.inference_cli` loads the base model in 8-bit, stitches in saved adapters, and exposes a Fire CLI. Paths can be passed via CLI flags or environment variables (`HIS_BASE_MODEL`, `HIS_LORA_WEIGHTS`, `HIS_SCALE_PATH`, `HIS_SCALES_DIR`).

- Specify the adapter checkpoint via `--lora_weights` (defaults to `models/1116_artist`). Pass a checkpoint such as `models/test_pkg_cuda0123` to load your latest finetune.
- Provide adapters to activate via `--active_adapter_names`. The script sanitizes the list: duplicates removed, opposing polarities trimmed, and professions restricted to zero/one selection. If you request nothing, it falls back to `["Artist"]`.
- Use `--cuda_devices` to pin inference to specific GPUs (e.g., `--cuda_devices 0`). Otherwise the driver respects `CUDA_VISIBLE_DEVICES`.

Example: **single adapter** (profession only):

```bash
python -m his.inference.inference_cli \
  --mode multi \
  --max_turns 1 \
  --user_prompt "What's your profession?" \
  --active_adapter_names '["Artist"]' \
  --lora_weights models/test_pkg_cuda0123 \
  --cuda_devices 0
```

Example: **multiple adapters** (profession + two traits):

```bash
python -m his.inference.inference_cli \
  --mode multi \
  --max_turns 1 \
  --user_prompt "Would you like to hang out this weekend?" \
  --active_adapter_names '["Doctor","EXT_low","AGR_high"]' \
  --lora_weights models/test_pkg_cuda0123 \
  --cuda_devices 0
```


---

## Evaluation

Three primary evaluation utilities now live under `his.evaluation`:

1. **Scale evaluations (`his.evaluation.scale_eval`)**  
   - Uses BF-marker-100 (20 prompts/trait) and the custom profession scale (20 prompts/profession).  
   - Implements an interactive questioning loop: ChatGPT asks the finetuned agent follow-up questions, then scores the response via self-consistency voting.

2. **Open-ended situational tests (`his.evaluation.situation_dialog`)**  
   - Eight scenarios mixing personalities and professions.  
   - Each dialogue runs for up to four turns; GPT judges detect which identities manifested.  
   - Experiments cover 971 identity combinations, mirroring the paper’s Section 4 results.

3. **Human/GPT evaluation helpers (`his.utils.gpt`, `his.evaluation.gpt_interviewer`)**  
   - Provide anonymized and reversed dialogues for human annotation or GPT-based judging.  
   - Require `OPENAI_API_KEY`; set `OPENAI_CHAT_MODEL`, `OPENAI_MAX_TOKENS` as desired.

Scale results and situational transcripts are saved under `scales/` by default. Manual evaluation templates live in `human_evaluation/`.

---

## Citation

If you use HIS or HIRPF in your work, please cite:

```
@article{sun2024identity,
  title={Identity-driven hierarchical role-playing agents},
  author={Sun, Libo and Wang, Siyuan and Huang, Xuanjing and Wei, Zhongyu},
  journal={arXiv preprint arXiv:2407.19412},
  year={2024}
}
```
