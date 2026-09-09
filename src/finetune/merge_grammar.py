"""
Merge grammar-derived pairs into the training set.

Reads grammar_pairs.jsonl (MP->FR) and appends them (both directions)
to the train split, skipping any pairs already present in train/val/test.
"""

import json
import re
from pathlib import Path

from config import (
    OUTPUT_DIR,
    OUTPUT_TRAIN,
    OUTPUT_VAL,
    OUTPUT_TEST,
    OUTPUT_GRAMMAR,
    MPO_LANGUAGE_CODE,
    FRENCH_CODE,
)


def normalize(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    if not path.exists():
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tr = rec.get("translation", rec)
            mp = tr.get(MPO_LANGUAGE_CODE) or tr.get("mpongwe", "")
            fr = tr.get(FRENCH_CODE) or tr.get("french", "")
            if mp and fr:
                pairs.append({"mpongwe": mp, "french": fr})
    return pairs


def main():
    existing = load_pairs(OUTPUT_TRAIN) + load_pairs(OUTPUT_VAL) + load_pairs(OUTPUT_TEST)
    seen = {normalize(p["mpongwe"]) + "\u0001" + normalize(p["french"]) for p in existing}
    print(f"Existing pairs (train+val+test lines): {len(existing)}")

    grammar = load_pairs(OUTPUT_GRAMMAR)
    print(f"Grammar pairs (MP->FR): {len(grammar)}")

    added = []
    skipped = 0
    for p in grammar:
        key = normalize(p["mpongwe"]) + "\u0001" + normalize(p["french"])
        rev_key = normalize(p["french"]) + "\u0001" + normalize(p["mpongwe"])
        if key in seen or rev_key in seen:
            skipped += 1
            continue
        seen.add(key)
        seen.add(rev_key)
        added.append(p)

    print(f"New pairs to add: {len(added)} (skipped {skipped} duplicates)")

    with open(OUTPUT_TRAIN, "a", encoding="utf-8") as f:
        for p in added:
            mp = p["mpongwe"]
            fr = p["french"]
            f.write(json.dumps({"translation": {MPO_LANGUAGE_CODE: mp, FRENCH_CODE: fr}}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"translation": {FRENCH_CODE: fr, MPO_LANGUAGE_CODE: mp}}, ensure_ascii=False) + "\n")

    total = sum(1 for _ in open(OUTPUT_TRAIN, encoding="utf-8"))
    print(f"Train split now: {total} lines")

    print("\nSample added pairs:")
    for p in added[:5]:
        print(f"  MP: {p['mpongwe']}")
        print(f"  FR: {p['french']}")
        print()


if __name__ == "__main__":
    main()