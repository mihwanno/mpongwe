"""
Prepare training data for NLLB QLoRA fine-tuning on Mpongwe translation.

Steps:
1. Extract sentence pairs (examples_usage) and definition pairs from the dictionary
2. Clean/filter/dedup
3. Include grammar-derived pairs (already produced by extract_grammar.py)
4. Shuffle, split train/val/test, write bidirectional NLLB JSONL

English is deferred (MP<->FR only, per project decision).
"""

import json
import random
import re
from pathlib import Path
from tqdm import tqdm

from config import (
    SOURCE_FILE,
    GRAMMAR_FILE,
    OUTPUT_TRAIN,
    OUTPUT_VAL,
    OUTPUT_TEST,
    OUTPUT_GRAMMAR,
    OUTPUT_AUGMENTED,
    MPO_LANGUAGE_CODE,
    FRENCH_CODE,
    VAL_SPLIT,
    TEST_SPLIT,
    MIN_PAIR_LENGTH,
    MAX_PAIR_LENGTH,
)

VALID_SOURCES = {
    "examples", "definition", "grammar_example", "grammar_verb",
    "grammar_noun", "grammar_possessive", "grammar_number",
    "grammar_adjective", "grammar_rule", "aug_conjug",
}


def load_dictionary() -> dict:
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace('"', "'")
    text = text.rstrip(".,; ")
    return text.strip()


def is_valid_pair(mp: str, fr: str) -> bool:
    if not mp or not fr:
        return False
    mp = clean_text(mp)
    fr = clean_text(fr)
    if len(mp) < MIN_PAIR_LENGTH or len(fr) < MIN_PAIR_LENGTH:
        return False
    if len(mp) > MAX_PAIR_LENGTH or len(fr) > MAX_PAIR_LENGTH:
        return False
    return True


def add_pair(pairs: list, seen: set, mp: str, fr: str, source: str, extra=None) -> bool:
    mp = clean_text(mp)
    fr = clean_text(fr)
    if not is_valid_pair(mp, fr):
        return False
    key = mp.lower() + "\u0001" + fr.lower()
    if key in seen:
        return False
    seen.add(key)
    pair = {"mpongwe": mp, "french": fr, "source": source}
    if extra:
        pair.update(extra)
    pairs.append(pair)
    return True


def extract_sentence_pairs(data: dict) -> list[dict]:
    pairs = []
    seen = set()
    for key, entry in tqdm(data.items(), desc="Extracting example pairs"):
        for ex in entry.get("examples_usage", []):
            for mp, fr in ex.items():
                add_pair(pairs, seen, mp, fr, "examples", {"headword": entry.get("mpongwe", key)})
    return pairs


def extract_definition_pairs(data: dict) -> list[dict]:
    pairs = []
    seen = set()
    for key, entry in tqdm(data.items(), desc="Extracting definition pairs"):
        mp = entry.get("mpongwe", key)
        fr = entry.get("francais", "")
        grammatical_type = entry.get("grammatical_type", "")
        if mp and fr:
            add_pair(pairs, seen, mp, fr, "definition",
                     {"grammatical_type": grammatical_type})
    return pairs


def load_grammar_pairs() -> list[dict]:
    """Load pairs already produced by extract_grammar.py."""
    pairs = []
    if not OUTPUT_GRAMMAR.exists():
        print(f"  (grammar_pairs.jsonl not found at {OUTPUT_GRAMMAR}; skipping)")
        return pairs
    with open(OUTPUT_GRAMMAR, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            mp = rec.get("mpongwe") or rec.get("translation", {}).get(MPO_LANGUAGE_CODE)
            fr = rec.get("french") or rec.get("translation", {}).get(FRENCH_CODE)
            source = rec.get("source", "grammar_rule")
            if mp and fr and source in VALID_SOURCES:
                pairs.append({"mpongwe": mp, "french": fr, "source": source})
    return pairs


def load_augmented_pairs() -> list[dict]:
    """Load morphologically-augmented verb pairs from augment_data.py."""
    pairs = []
    if not OUTPUT_AUGMENTED.exists():
        print(f"  (augmented_pairs.jsonl not found at {OUTPUT_AUGMENTED}; skipping)")
        return pairs
    with open(OUTPUT_AUGMENTED, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            mp = rec.get("mpongwe")
            fr = rec.get("french")
            if mp and fr:
                pairs.append({"mpongwe": mp, "french": fr, "source": "aug_conjug"})
    return pairs


def split_data(pairs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    random.seed(42)
    random.shuffle(pairs)
    n = len(pairs)
    n_test = max(1, int(n * TEST_SPLIT))
    n_val = max(1, int(n * VAL_SPLIT))
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]
    return train, val, test


def write_nllb_format(pairs: list[dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps({"translation": {
                MPO_LANGUAGE_CODE: pair["mpongwe"],
                FRENCH_CODE: pair["french"],
            }}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"translation": {
                FRENCH_CODE: pair["french"],
                MPO_LANGUAGE_CODE: pair["mpongwe"],
            }}, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("Mpongwe Translation Data Preparation")
    print("=" * 60)

    data = load_dictionary()
    print(f"Loaded {len(data):,} dictionary entries")

    sentence_pairs = extract_sentence_pairs(data)
    print(f"Example sentence pairs: {len(sentence_pairs):,}")

    definition_pairs = extract_definition_pairs(data)
    print(f"Definition pairs: {len(definition_pairs):,}")

    grammar_pairs = load_grammar_pairs()
    print(f"Grammar-derived pairs: {len(grammar_pairs):,}")

    augmented_pairs = load_augmented_pairs()
    print(f"Augmented verb pairs: {len(augmented_pairs):,}")

    all_pairs = sentence_pairs + definition_pairs + grammar_pairs + augmented_pairs

    # Global dedup across all sources
    seen = set()
    unique = []
    for p in all_pairs:
        key = p["mpongwe"].lower() + "\u0001" + p["french"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    print(f"After global dedup: {len(unique):,}")

    train, val, test = split_data(unique)
    print(f"Split: train={len(train):,} val={len(val):,} test={len(test):,}")

    write_nllb_format(train, OUTPUT_TRAIN)
    write_nllb_format(val, OUTPUT_VAL)
    write_nllb_format(test, OUTPUT_TEST)

    print(f"\nOutput files (bidirectional, {2} lines per pair):")
    for p in (OUTPUT_TRAIN, OUTPUT_VAL, OUTPUT_TEST):
        n = sum(1 for _ in open(p, encoding="utf-8"))
        print(f"  {p}: {n:,} lines")

    print("\nSample pairs:")
    for pair in train[:5]:
        print(f"  MP: {pair['mpongwe']}")
        print(f"  FR: {pair['french']}  [{pair['source']}]")


if __name__ == "__main__":
    main()