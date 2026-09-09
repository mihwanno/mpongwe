"""
Evaluate a fine-tuned Mpongwe translation model on the test set.

Metrics (mirrors README's evaluation strategy):
  - chrF++  (primary): character n-gram F-score, best for morphologically rich langs
  - BLEU    (secondary): standard corpus BLEU (sacrebleu, 13a tokenization)
  - COMET   (ranking, optional): learned metric; needs `pip install unbabel-comet`

Usage:
    python evaluate.py --model_path data/finetune/checkpoints/merged \
                       --test_file data/finetune/test.jsonl
    python evaluate.py --model_path ... --with-comet --comet-bs 16
"""

import argparse
import json
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from config import (
    OUTPUT_TEST,
    MPO_LANGUAGE_CODE,
    FRENCH_CODE,
)


def load_test_rows(path: Path) -> tuple[list[str], list[str]]:
    """Return (sources, targets) for the MP->FR direction of the test set.

    The test file interleaves both directions (MP->FR then FR->MP). We select
    MP->FR lines by checking which language key comes first.
    """
    sources, targets = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            tr = rec.get("translation", rec)
            if not tr:
                continue
            first_key = next(iter(tr.keys()))
            if first_key == MPO_LANGUAGE_CODE:
                sources.append(tr[MPO_LANGUAGE_CODE])
                targets.append(tr[FRENCH_CODE])
    return sources, targets


def translate_batch(model, tokenizer, texts: list[str], src_lang: str, tgt_lang: str,
                    max_length: int = 256, batch_size: int = 32, num_beams: int = 4):
    tokenizer.src_lang = src_lang
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    outs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"{src_lang}->{tgt_lang}"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_length=max_length,
                num_beams=num_beams,
            )
        outs.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return [o.strip() for o in outs]


def score(preds: list[str], refs: list[str]) -> dict:
    refs_list = [[r] for r in refs]
    bleu = sacrebleu.corpus_bleu(preds, refs_list)
    chrf = sacrebleu.corpus_chrf(preds, refs_list,
                                 char_order=6, word_order=2,
                                 beta=2, remove_whitespace=True)
    return {"bleu": bleu.score, "chrf": chrf.score}


def compute_comet(preds, srcs, refs, bs):
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        raise RuntimeError("COMET needs: pip install unbabel-comet")

    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
    model_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(model_path)
    outs = comet_model.predict(data, batch_size=bs, progress_bar=True)
    return outs["system_score"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Mpongwe translation model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, default=str(OUTPUT_TEST))
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--with_comet", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_path = Path(args.test_file)
    print(f"Loading test set: {test_path}")
    srcs, refs = load_test_rows(test_path)
    print(f"Test pairs: {len(refs)}")

    print("Translating Mpongwe -> French...")
    preds = translate_batch(model, tokenizer, srcs, MPO_LANGUAGE_CODE, FRENCH_CODE,
                            args.max_length, args.batch_size, args.num_beams)

    print("Scoring...")
    m = score(preds, refs)
    print(f"  BLEU : {m['bleu']:.2f}")
    print(f"  chrF++: {m['chrf']:.2f}")

    if args.with_comet:
        print("Computing COMET (this downloads a model)...")
        m["comet"] = compute_comet(preds, srcs, refs, args.batch_size)
        print(f"  COMET: {m['comet']:.4f}")

    print("\nSample predictions:")
    for i in range(min(5, len(srcs))):
        print(f"  SRC: {srcs[i]}")
        print(f"  REF: {refs[i]}")
        print(f"  MT : {preds[i]}")
        print()


if __name__ == "__main__":
    main()