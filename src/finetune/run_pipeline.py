"""
Mpongwe QLoRA fine-tuning pipeline.

Run end-to-end or stage-by-stage:

    # Full pipeline
    python run_pipeline.py --epochs 3 --batch_size 8

    # Single stage
    python run_pipeline.py --only train --epochs 3 --batch_size 8
    python run_pipeline.py --only evaluate
    python run_pipeline.py --only inference --interactive

Stage order:
    1. install_dependencies  (transformers, peft, bitsandbytes, ...)
    2. add_token.py          (add mpo_Latn to tokenizer + base embeddings)
    3. prepare_data.py       (build train/val/test from dictionary + grammar + aug)
    4. train.py              (QLoRA fine-tune, saves adapter + merged model)
    5. evaluate.py           (BLEU / chrF++ / COMET on the merged model)
    6. inference.py          (interactive demo)
"""

import argparse
import subprocess
import sys
from pathlib import Path

FINETUNE_DIR = Path(__file__).parent


def install_dependencies(test: bool = False):
    packages = [
        "transformers>=4.40.0",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "sacrebleu",
        "sentencepiece",
        "numpy",
        "scipy",
    ]
    if test:
        packages += ["evaluate", "pytest"]
    for pkg in packages:
        print(f"  pip install {pkg}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", pkg], check=True
        )
    print("Dependencies installed.")


def run_stage(name: str, *args: str):
    print(f"\n{'=' * 60}\nStage: {name}\n{'=' * 60}")
    cmd = [sys.executable, str(FINETUNE_DIR / f"{name}.py"), *args]
    print("> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Mpongwe QLoRA pipeline")
    parser.add_argument("--base_model", type=str,
                        default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--output_dir", type=str,
                        default=str(FINETUNE_DIR.parent.parent / "data" / "finetune" / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--only", type=str, default=None,
                        help="Run a single stage (install/add_token/prepare/train/evaluate/inference)")
    args = parser.parse_args()

    if not args.skip_install:
        install_dependencies()

    base = Path(args.output_dir) / "base"
    if args.only in (None, "add_token", "train", "evaluate", "inference"):
        run_stage("add_token", "--model_name", args.base_model, "--output_dir", str(base))
    if args.only in (None, "prepare"):
        run_stage("prepare_data")
    if args.only in (None, "train"):
        run_stage(
            "train",
            "--model_dir", str(base),
            "--output_dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--eval_steps", str(args.eval_steps),
            "--save_steps", str(args.save_steps),
            "--max_length", str(args.max_length),
        )
    if args.only in (None, "evaluate"):
        run_stage("evaluate", "--model_path", str(Path(args.output_dir) / "merged"))
    if args.only in (None, "inference"):
        run_stage("inference", "--model_path", str(Path(args.output_dir) / "merged"),
                  "--interactive")


if __name__ == "__main__":
    main()
