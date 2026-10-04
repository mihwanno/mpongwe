# Mpongwe ↔ French Translation Model (NLLB-200 QLoRA Fine-tuning)

Fine-tuning Meta's `facebook/nllb-200-distilled-600M` for Mpongwe (Bantu, Gabon) using QLoRA. Trained on a dictionary (Raponda-Walker), grammar examples, and rule-based verb conjugation augmentations.

## Training in Lightning AI Studio

Lightning AI Studio (https://lightning.ai/studio) is recommended for training (T4/L4 GPUs). The codebase is optimized to run QLoRA fine-tuning on a single T4 (~4–8 GB VRAM).

### 1. Create a Lightning AI Studio
- Go to https://lightning.ai/studio
- Create a new Studio (choose a GPU: T4 is sufficient; L4 faster)
- Clone this repository: `git clone https://github.com/<your-org>/mpongwe.git && cd mpongwe`

### 2. Set up environment

Recommended (uses `uv`, installs from `requirements.txt`, pins Python 3.12).
Do NOT use `uv sync` — it also resolves the optional `comet` extra, which
conflicts (`unbabel-comet` needs `numpy<2`, `langchain-community==0.4.1`
needs `numpy>=2.1`).

In Studio terminal:

```bash
chmod +x setup_env.sh
./setup_env.sh
source .venv/bin/activate
python --version  # expect 3.12.x
```

Manual fallback (same result, no `uv sync`):

```bash
rm -rf .venv
uv python install 3.12
uv venv --seed --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
# or: pip install -r requirements.txt
```

COMET is optional and excluded by default (see conflict above). Only for
COMET eval, preferably in a separate env:

```bash
pip install "unbabel-comet>=2.0.2" "numpy<2"
```

#### Troubleshooting

1. `TOML parse error at line 32 ... requires-python ... expected a sequence`
   - Cause: old clone has `requires-python = ">=3.12"` at the bottom of
     `pyproject.toml`, under `[project.optional-dependencies]` (values there
     must be lists). It belongs under `[project]` (line ~6).
   - Fix: `git pull`, then `grep -n requires-python pyproject.toml` must show
     line 6. If you hand-edited the file and the line is gone, restore it:
     ```bash
     python3 -c "
     t = open('pyproject.toml').read()
     assert 'requires-python' not in t
     t = t.replace('readme = \"README.md\"', 'readme = \"README.md\"\nrequires-python = \">=3.12\"')
     open('pyproject.toml','w').write(t)
     "
     ```
     Then `rm -rf .venv` and re-run `./setup_env.sh`.

2. `No requires-python value found ... Defaulting to >=3.14`
   - Cause: `requires-python` missing (see fix above). Without it uv picks
     3.14, which lacks `faiss-cpu` / `bitsandbytes` / `torch` wheels.
   - Fix: restore the line, then `uv python install 3.12 && rm -rf .venv && ./setup_env.sh`.

3. `No solution found ... unbabel-comet ... numpy`
   - Cause: resolving the `comet` extra together with the default deps.
   - Fix: don't use `uv sync` / don't request the `comet` extra. `setup_env.sh`
     already installs from `requirements.txt`, which excludes COMET.

4. `/path/.venv/bin/python: No module named pip` in `run_pipeline.py`
   - Cause: venv created with plain `uv venv` (no pip) and/or still on
     Python 3.14 (see `cpython-3.14.5` in the traceback). `setup_env.sh` now
     uses `uv venv --seed --python 3.12`; old `.venv` dirs lack pip.
   - Fix: pull latest, recreate the env, then skip the pipeline's
     re-install (deps are already installed):
     ```bash
     git pull
     rm -rf .venv
     chmod +x setup_env.sh && ./setup_env.sh
     source .venv/bin/activate
     python --version  # expect 3.12.x
     python -m pip --version  # must work
     python src/finetune/run_pipeline.py --skip_install \
       --epochs 3 --batch_size 8 --eval_steps 500 --save_steps 500 --max_length 256
     ```
     `run_pipeline.py` now also prefers `uv pip install` and bootstraps pip
     via `ensurepip` if missing, so plain `--epochs ...` works too.

5. `FileNotFoundError: data/processed/mpongwe_francais.json` in `prepare_data`
   - Cause: `data/processed/` is git-ignored, so clones lack the source
     dictionary (built locally by `src/digitize_raponda/`
     `make_complete_dictionary.py`, also ignored). The shipped
     `data/finetune/train|val|test.jsonl` are tracked and sufficient.
   - Fix: pull latest and re-run — full pipeline auto-skips `prepare_data`
     when outputs exist; or force-skip explicitly:
     ```bash
     git pull
     ls -lh data/finetune/  # train.jsonl, val.jsonl, test.jsonl must exist
     python src/finetune/run_pipeline.py --skip_install --skip_prepare \
       --epochs 3 --batch_size 8 --eval_steps 500 --save_steps 500 --max_length 256
     # Rebuild only if you have data/processed/:
     # python src/finetune/prepare_data.py --force
     ```

### 3. Verify data exists

Processed training data should already exist:
```bash
ls -lh data/finetune/
# train.jsonl, val.jsonl, test.jsonl, grammar_pairs.jsonl, augmented_pairs.jsonl
```

A token-augmented base model (with `mpo_Latn`) may already exist at:
```bash
ls -lh data/finetune/checkpoints/base/
```

If missing, the pipeline will generate it (downloads NLLB-200-distilled-600M and adds the `mpo_Latn` token).

### 4. Train with the pipeline (recommended)

Run the full end-to-end QLoRA fine-tuning pipeline (stages: add_token → prepare_data → train → evaluate → inference).
`prepare_data` auto-skips when `data/finetune/train|val|test.jsonl` are already
shipped and `data/processed/mpongwe_francais.json` is missing (it is
git-ignored, so fresh clones don't have it):

```bash
python src/finetune/run_pipeline.py --skip_install \
  --epochs 3 \
  --batch_size 8 \
  --eval_steps 500 \
  --save_steps 500 \
  --max_length 256
```

To skip dependency installation (if already installed):
```bash
python src/finetune/run_pipeline.py --skip_install \
  --epochs 3 --batch_size 8 --eval_steps 500 --save_steps 500 --max_length 256
```

To resume from the latest checkpoint or run a single stage:
```bash
# Resume training only
python src/finetune/run_pipeline.py --only train \
  --epochs 3 --batch_size 8 --eval_steps 500 --save_steps 500 --max_length 256

# Evaluate merged model
python src/finetune/run_pipeline.py --only evaluate

# Interactive inference with merged model
python src/finetune/run_pipeline.py --only inference
```

### 5. Training outputs

Artifacts are saved under `data/finetune/checkpoints/`:
- `base/` — Token-augmented NLLB base model (mpo_Latn added)
- `checkpoint-*/` — HF Trainer checkpoints (saved per `save_steps`)
- `adapter/` — LoRA adapter weights (PEFT)
- `merged/` — Base + LoRA merged model (ready for inference/eval)

Logs: Trainer prints metrics (BLEU, chrF++) at each eval step; best model tracked by `chrf`.

### 6. Direct training (advanced)

You can also call `train.py` directly if you want finer control:

```bash
python src/finetune/train.py \
  --model_dir data/finetune/checkpoints/base \
  --output_dir data/finetune/checkpoints \
  --epochs 3 \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 2 \
  --warmup_steps 100 \
  --eval_steps 500 \
  --save_steps 500 \
  --max_length 256
```

Add `--cpu` only for local debugging (disables 4-bit QLoRA, uses fp32).

### 7. Evaluation

Evaluate on the test set (BLEU + chrF++). Optionally include COMET (requires `unbabel-comet`):

```bash
python src/finetune/evaluate.py --model_path data/finetune/checkpoints/merged
python src/finetune/evaluate.py --model_path data/finetune/checkpoints/merged --with_comet
```

### 8. Inference

Interactive translation (auto-detects MP→FR vs FR→MP direction):
```bash
python src/finetune/inference.py --model_path data/finetune/checkpoints/merged --interactive
```

Single translation:
```bash
python src/finetune/inference.py --model_path data/finetune/checkpoints/merged \
  --text "Pusi yi daga"
```

## Notes on Lightning AI Studio

- Training typically completes in ~30–90 minutes on T4 depending on epochs/steps.
- Checkpoints are written to disk; Studio volumes persist across sessions if configured.
- If you see bitsandbytes/accelerate import issues, ensure GPU drivers are available (Studio provides CUDA). `run_pipeline.py` will auto-install required packages.
- The pipeline re-runs `add_token` if called with `--only train/evaluate/inference`; this is safe (reuses cached base if unchanged).

## Project structure (training)

- `src/finetune/config.py` — Paths, language codes, splits
- `src/finetune/add_token.py` — Add `mpo_Latn` token to NLLB tokenizer
- `src/finetune/prepare_data.py` — Build bidirectional train/val/test JSONL
- `src/finetune/train.py` — QLoRA fine-tune (PEFT + bitsandbytes 4-bit nf4)
- `src/finetune/evaluate.py` — BLEU/chrF++ (+COMET optional)
- `src/finetune/inference.py` — Translator CLI
- `src/finetune/run_pipeline.py` — Orchestrator entry point
- `src/finetune/extract_grammar.py`, `augment_data.py`, `merge_grammar.py` — Data synthesis scripts (not wired into pipeline)

## Dependencies

Core training deps (declared in `requirements.txt`): transformers>=4.40.0, datasets, accelerate, peft, bitsandbytes, sacrebleu, sentencepiece, numpy, scipy, evaluate.

See `pyproject.toml` for full project deps (includes Streamlit app stack).
