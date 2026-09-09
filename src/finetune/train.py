"""
Fine-tune NLLB-200 on Mpongwe translation using QLoRA.

Designed for Google Colab free tier (T4 GPU, 16GB VRAM).

Prereq: run add_token.py first to add the mpo_Latn token, OR the base dir
already contains it. 4-bit QLoRA keeps ~4-5GB of VRAM for the 600M model.

Usage:
    python add_token.py --model_name facebook/nllb-200-distilled-600M
    python train.py --model_dir data/finetune/nllb-mpongwe-base --output_dir data/finetune/checkpoints
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import sacrebleu
import numpy as np

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftConfig,
        PeftModel,
    )
    _PEFT_OK = True
except ImportError:
    _PEFT_OK = False

from config import (
    OUTPUT_DIR,
    OUTPUT_TRAIN,
    OUTPUT_VAL,
    MPO_LANGUAGE_CODE,
    FRENCH_CODE,
)

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tr = rec.get("translation", rec)
            if MPO_LANGUAGE_CODE in tr and FRENCH_CODE in tr:
                rows.append(
                    {
                        "src": tr[MPO_LANGUAGE_CODE],
                        "tgt": tr[FRENCH_CODE],
                        "src_lang": MPO_LANGUAGE_CODE,
                        "tgt_lang": FRENCH_CODE,
                    }
                )
            elif FRENCH_CODE in tr and MPO_LANGUAGE_CODE in tr:
                rows.append(
                    {
                        "src": tr[FRENCH_CODE],
                        "tgt": tr[MPO_LANGUAGE_CODE],
                        "src_lang": FRENCH_CODE,
                        "tgt_lang": MPO_LANGUAGE_CODE,
                    }
                )
    return rows


def make_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    """Tokenize rows grouped by (src_lang, tgt_lang) so both are set correctly."""
    ds = Dataset.from_list(rows)
    directions = {(r["src_lang"], r["tgt_lang"]) for r in rows}
    tokenized = []
    for src, tgt in directions:
        sub = ds.filter(lambda r: r["src_lang"] == src and r["tgt_lang"] == tgt)
        tokenizer.src_lang = src
        tokenizer.tgt_lang = tgt
        cols = sub.map(
            lambda batch: tokenize_batch(batch, tokenizer, max_length),
            batched=True,
            remove_columns=sub.column_names,
        )
        tokenized.append(cols)
    return concatenate_datasets(tokenized)


def tokenize_batch(batch: dict, tokenizer, max_length: int) -> dict:
    model_inputs = tokenizer(
        list(batch["src"]),
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        text_target=list(batch["tgt"]),
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    model_inputs["labels"] = [
        [tok if tok != tokenizer.pad_token_id else -100 for tok in seq]
        for seq in labels["input_ids"]
    ]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = [p.strip().lower() for p in decoded_preds]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [l.strip().lower() for l in decoded_labels]
    decoded_labels = [[ref] for ref in decoded_labels]

    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels)
    chrf = sacrebleu.corpus_chrf(decoded_preds, decoded_labels)
    return {"bleu": bleu.score, "chrf": chrf.score}


def build_qlora_model(tokenizer, model_dir: str, cpu: bool = False):
    if cpu:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        print("CPU debug mode: 4-bit quantization disabled, running in fp32")
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir, quantization_config=bnb_config, device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def load_model_for_train(model_dir: str, cpu: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if MPO_LANGUAGE_CODE not in tokenizer.get_vocab():
        raise RuntimeError(
            f"'{MPO_LANGUAGE_CODE}' not in tokenizer. Run add_token.py first."
        )
    if not _PEFT_OK:
        raise RuntimeError("peft is required for QLoRA. Run: pip install peft")
    model = build_qlora_model(tokenizer, model_dir, cpu=cpu)
    model.print_trainable_parameters()
    device = model.device
    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return tokenizer, model


def test_translations(model, tokenizer, n_samples: int = 5):
    test_pairs = [
        ("Pusi yi daga", "le chat miaule"),
        ("O re mbato g'ozógè", "tu es avare comme un plant de dracéna"),
        ("Ntee yíno yi sulaga ni vunga", "il y a toujours des épidémies dans ce pays"),
        ("Othwana ompolo", "un grand enfant"),
    ]

    mpo_id = tokenizer.convert_tokens_to_ids(MPO_LANGUAGE_CODE)
    fr_id = tokenizer.convert_tokens_to_ids(FRENCH_CODE)

    print("\nMpongwe → French:")
    for mp, expected_fr in test_pairs:
        tokenizer.src_lang = MPO_LANGUAGE_CODE
        tokenizer.tgt_lang = FRENCH_CODE
        inputs = tokenizer(mp, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, forced_bos_token_id=fr_id, max_length=256, num_beams=3
            )
        pred_fr = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  MP: {mp}")
        print(f"  Expected: {expected_fr}")
        print(f"  Predicted: {pred_fr}")

    print("\nFrench → Mpongwe:")
    for mp, fr in test_pairs:
        tokenizer.src_lang = FRENCH_CODE
        tokenizer.tgt_lang = MPO_LANGUAGE_CODE
        inputs = tokenizer(fr, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, forced_bos_token_id=mpo_id, max_length=256, num_beams=3
            )
        pred_mp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  FR: {fr}")
        print(f"  Expected: {mp}")
        print(f"  Predicted: {pred_mp}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune NLLB on Mpongwe")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Base model dir with mpo_Latn token (run add_token.py first)")
    parser.add_argument("--output_dir", type=str,
                        default=str(OUTPUT_DIR / "checkpoints"))
    parser.add_argument("--train_file", type=str, default=str(OUTPUT_TRAIN))
    parser.add_argument("--val_file", type=str, default=str(OUTPUT_VAL))
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="LoRA adapter LR (higher than full FT LR is standard)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--cpu", action="store_true",
                        help="Debug on CPU: no 4-bit quantization, fp32, LoRA instead of QLoRA")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training rows (debug)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit validation rows (debug)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Limit total training steps (debug)")
    args = parser.parse_args()

    print("=" * 60)
    print("NLLB Mpongwe QLoRA Fine-tuning")
    print("=" * 60)

    tokenizer, model = load_model_for_train(args.model_dir, cpu=args.cpu)

    print(f"Loading datasets from {args.train_file} / {args.val_file}...")
    train_rows = load_rows(Path(args.train_file))
    val_rows = load_rows(Path(args.val_file))
    if args.max_train_samples:
        train_rows = train_rows[:args.max_train_samples]
    if args.max_val_samples:
        val_rows = val_rows[:args.max_val_samples]
    print(f"Train rows: {len(train_rows)} ({len(train_rows)//2} pairs), "
          f"Val rows: {len(val_rows)} ({len(val_rows)//2} pairs)")

    train_dataset = make_dataset(train_rows, tokenizer, args.max_length)
    val_dataset = make_dataset(val_rows, tokenizer, args.max_length)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100
    )

    training_kwargs = dict(
        output_dir=args.output_dir,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=min(args.batch_size, 8),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=2,
        fp16=not args.cpu,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        report_to="none",
    )
    if args.max_steps:
        training_kwargs["max_steps"] = args.max_steps
    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    checkpoint_dir = Path(args.output_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
    if checkpoints:
        latest = str(checkpoints[-1])
        print(f"\nResuming from checkpoint: {latest}")
        trainer.train(resume_from_checkpoint=latest)
    else:
        print(f"\nStarting training from scratch...")
        trainer.train()

    # Save adapter
    adapter_dir = Path(args.output_dir) / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"LoRA adapter saved to: {adapter_dir}")

    # Merge and save full model for CPU/inference
    print("Merging LoRA adapter into base model...")
    merged = trainer.model.merge_and_unload()
    merged_dir = Path(args.output_dir) / "merged"
    merged.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    print(f"Merged model saved to: {merged_dir}")

    print("\nRunning sample translations...")
    test_translations(merged, tokenizer)


if __name__ == "__main__":
    main()