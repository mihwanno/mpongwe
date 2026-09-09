"""
Morphological augmentation: conjugate dictionary verbs using grammar-derived
rules (a model-free alternative to back-translation, which can't run until we
have a trained model).

For each dictionary verb entry we generate Mpongwe <-> French pairs in
present / present-negative / future / future-negative / conditional
across 6 persons, using:
  - Mpongwe: pronoun + stem templates from the grammar (kamba paradigm)
  - French : a conservative rule-based conjugator (mostly 1st-group -er
             verbs, 2nd-group -ir, reflexives, and a short irregular table).

We only emit clean, confident forms; anything not safely classifiable is
skipped so we never feed noisy labels to the model.

Run AFTER extract_grammar.py, BEFORE prepare_data.py.
"""

import json
import re
from pathlib import Path

from config import SOURCE_FILE, OUTPUT_DIR

OUTPUT_FILE = OUTPUT_DIR / "augmented_pairs.jsonl"

# ---------------------------------------------------------------------------
# French conjugation (conservative)
# ---------------------------------------------------------------------------

# Full irregulars we handle explicitly: {infinitive: 3sg "il" form + ...}
# We store the full 6-form present and future.
FR_IRREGULAR = {
    "aller": {"present": ["vais", "vas", "va", "allons", "allez", "vont"],
              "future": ["irai", "iras", "ira", "irons", "irez", "iront"]},
    "être": {"present": ["suis", "es", "est", "sommes", "êtes", "sont"],
             "future": ["serai", "seras", "sera", "serons", "serez", "seront"]},
    "avoir": {"present": ["ai", "as", "a", "avons", "avez", "ont"],
              "future": ["aurai", "auras", "aura", "aurons", "aurez", "auront"]},
    "faire": {"present": ["fais", "fais", "fait", "faisons", "faites", "font"],
              "future": ["ferai", "feras", "fera", "ferons", "ferez", "feront"]},
    "dire": {"present": ["dis", "dis", "dit", "disons", "dites", "disent"],
             "future": ["dirai", "diras", "dira", "dirons", "direz", "diront"]},
    "voir": {"present": ["vois", "vois", "voit", "voyons", "voyez", "voient"],
             "future": ["verrai", "verras", "verra", "verrons", "verrez", "verront"]},
    "venir": {"present": ["viens", "viens", "vient", "venons", "venez", "viennent"],
              "future": ["viendrai", "viendras", "viendra", "viendrons", "viendrez", "viendront"]},
    "prendre": {"present": ["prends", "prends", "prend", "prenons", "prenez", "prennent"],
                "future": ["prendrai", "prendras", "prendra", "prendrons", "prendrez", "prendront"]},
    "mettre": {"present": ["mets", "mets", "met", "mettons", "mettez", "mettent"],
               "future": ["mettrai", "mettras", "mettra", "mettrons", "mettrez", "mettront"]},
    "pouvoir": {"present": ["peux", "peux", "peut", "pouvons", "pouvez", "peuvent"],
                "future": ["pourrai", "pourras", "pourra", "pourrons", "pourrez", "pourront"]},
    "vouloir": {"present": ["veux", "veux", "veut", "voulons", "voulez", "veulent"],
                "future": ["voudrai", "voudras", "voudra", "voudrons", "voudrez", "voudront"]},
}

# -ir verbs that are 3rd-group (irregular): we skip these.
IRREGULAR_IR = {
    "partir", "dormir", "mourir", "courir", "sentir", "sortir", "servir",
    "mentir", "ouvrir", "offrir", "souffrir", "cueillir", "fuir", "bouillir",
    "se souvenir", "se repentir", "se tenir", "tenir", "couvrir", "vêtir",
    "obtenir", "détenir", "retenir", "appartenir", "maintenir", "contenir",
    "soutenir", "venir", "devenir", "revenir", "prévenir", "convenir",
    "intervenir", "parvenir", "survenir", "acquérir", "faillir",
}

# Verbs with a double-consonant / stem-vowel shift in the present (-eler/-eter).
DOUBLE_CONSONANT = {
    "appeler", "rappeler", "jeter", "rejeter", "projeter", "s'appeler",
    "feuilleter", "projeter", "épeler",
}
STEM_VOWEL_SHIFT = {  # -e_er -> -è_er, -é_er -> -è_er (je/tu/il/ils)
    "acheter", "racheter", "peser", "mener", "amener", "emmener", "lever",
    "soulever", "enlever", "élever", "semer", "crever", "prélever", "préférer",
    "espérer", "répéter", "posséder", "céder",
}


def _vowel_start(s: str) -> bool:
    return len(s) > 0 and s[0] in "aeiouàâéèêîôûh"


def elide_je(form: str) -> str:
    return f"j'{form}" if _vowel_start(form) else f"je {form}"


SUBJ = ["je", "tu", "il", "nous", "vous", "ils"]


def _with_subject(verb: str, person: int) -> str:
    if person == 0:
        return elide_je(verb)
    return f"{SUBJ[person]} {verb}"


def _conj_fut(inf: str) -> list[str]:
    """Future of a regular verb: stem = full infinitive, endings ai/as/a/ons/ez/ont."""
    return [_with_subject(inf + e, i)
            for i, e in enumerate(["ai", "as", "a", "ons", "ez", "ont"])]


def _conj_er(inf: str) -> list[str]:
    """Present tense of a 1st-group verb, with subject pronouns."""
    stem = inf[:-2]
    stem_stressed = re.sub(r"[eé]([bcdfghjklmnpqrstvwxz]+)$", r"è\1", stem)

    present = []
    for i, ending in enumerate(["e", "es", "e", "ons", "ez", "ent"]):
        if inf in DOUBLE_CONSONANT and i in (0, 1, 2, 5):
            base = stem + stem[-1]
        elif inf in STEM_VOWEL_SHIFT and i in (0, 1, 2, 5):
            base = stem_stressed
        else:
            base = stem
        form = base + ending
        if inf.endswith("cer") and ending in ("ons", "ez"):
            form = base[:-1] + "ç" + ending
        elif inf.endswith("ger") and ending == "ons":
            form = base + "eons"
        present.append(_with_subject(form, i))
    return present


def _wrap_reflexive(moods: dict, reflexive: bool) -> dict:
    """Prefix reflexive pronouns onto 6-form moods: [je, tu, il, nous, vous, ils]."""
    if not reflexive:
        return moods
    pro_ref = ["me", "te", "se", "nous", "vous", "se"]
    out = {}
    for tense, forms in moods.items():
        res = []
        for i, form in enumerate(forms):
            verb = form.split(" ", 1)[-1]
            if verb.startswith("j'"):
                verb = verb[2:]
            ref = pro_ref[i]
            sep = " "
            if _vowel_start(verb) and ref in ("me", "te", "se"):
                ref = ref[0] + "'"
                sep = ""
            res.append(f"{SUBJ[i]} {ref}{sep}{verb}")
        out[tense] = res
    return out


def conjugate_french(inf: str) -> dict | None:
    """Return {'present': [6], 'future': [6]} or None if not safely conjugable."""
    reflexive = False
    if inf.startswith("se "):
        reflexive, inf = True, inf[3:].strip()
    elif inf.startswith("s'"):
        reflexive, inf = True, inf[2:].strip()

    # Causative gloss "faire <inf>" -> conjugate faire and append the infinitive
    if inf.startswith("faire ") and not reflexive:
        obj = inf[len("faire "):].strip()
        base = FR_IRREGULAR["faire"]
        present = [_with_subject(f, i) + f" {obj}" for i, f in enumerate(base["present"])]
        future = [_with_subject(f, i) + f" {obj}" for i, f in enumerate(base["future"])]
        return {"present": present, "future": future}

    if inf in FR_IRREGULAR:
        present = [_with_subject(f, i) for i, f in enumerate(FR_IRREGULAR[inf]["present"])]
        future = [_with_subject(f, i) for i, f in enumerate(FR_IRREGULAR[inf]["future"])]
        return _wrap_reflexive({"present": present, "future": future}, reflexive)

    if inf.endswith("er"):
        return _wrap_reflexive({"present": _conj_er(inf), "future": _conj_fut(inf)}, reflexive)

    if inf.endswith("ir"):
        if inf in IRREGULAR_IR:
            return None
        stem = inf[:-2]
        present = [_with_subject(stem + e, i)
                   for i, e in enumerate(["is", "is", "it", "issons", "issez", "issent"])]
        return _wrap_reflexive({"present": present, "future": _conj_fut(inf)}, reflexive)

    return None


# ---------------------------------------------------------------------------
# Mpongwe conjugation (from grammar: kamba paradigm, strong stem = headword)
# ---------------------------------------------------------------------------

MP_SUBJECTS = [  # (pronoun, french pronoun)
    ("mi", "je"),
    ("o", "tu"),
    ("e", "il"),
    ("azwe", "nous"),
    ("anwe", "vous"),
    ("wi", "ils"),
]

MPO_TEMPLATES = {  # pronoun + marker + stem
    "present": "",
    "present_neg": "pa ",
    "future": "be ",
    "future_neg": "'be ",
    "conditional": "to ",
}


def conjugate_mpongwe(stem: str) -> dict[str, list[str]]:
    """Return {tense: [6 forms]} using grammar pronoun templates (kamba paradigm)."""
    out = {tense: [] for tense in MPO_TEMPLATES}
    for subj, _ in MP_SUBJECTS:
        out["present"].append(f"{subj} {stem}")
        out["present_neg"].append(f"{subj} pa {stem}")
        out["future"].append(f"{subj} be {stem}")
        out["future_neg"].append(f"{subj}'be {stem}")
        out["conditional"].append(f"{subj} to {stem}")
    out["future_neg"][0] = f"m'be {stem}"  # mi + 'be -> m'be
    return out


# ---------------------------------------------------------------------------
# Pair emission
# ---------------------------------------------------------------------------

def first_french_infinitive(fr: str) -> str | None:
    """Extract first clean French infinitive from a dictionary gloss."""
    fr = re.sub(r"\(.*?\)", "", fr)
    first = fr.split(";")[0].split(",")[0].strip()
    first = first.replace("…", "").replace("...", "").strip()
    if not first or len(first) < 2:
        return None
    if first.startswith(("se ", "s'", "faire ")):
        base = first.split(" ", 1)[-1].lstrip("'")
        return first if re.search(r"(er|ir|re|oir)$", base) else None
    if " " in first:
        return None
    return first if re.search(r"(er|ir|re|oir)$", first) else None


def build_verb_list(data: dict) -> list[dict]:
    verbs = []
    for key, entry in data.items():
        gt = str(entry.get("grammatical_type", "")).strip().lower()
        if not gt.startswith("v."):
            continue
        mp = re.sub(r"\s*\(.*?\)", "", entry.get("mpongwe", key)).strip()
        fr = entry.get("francais", "")
        verbs.append({"mp": mp, "fr": fr, "type": gt})
    return verbs


def negate_french(form: str) -> str:
    """Wrap an already-conjugated French form into the negative (ne ... pas)."""
    # order matters: match longer prefixes first ("ils" before "il")
    for subj, new_subj in [("j'", "je"), ("je", "je"), ("ils", "ils"),
                          ("il", "il"), ("nous", "nous"), ("vous", "vous"),
                          ("tu", "tu")]:
        if form.startswith(subj):
            rest = form[len(subj):].strip()
            n = "n'" if rest and _vowel_start(rest) else "ne"
            sep = "" if n.endswith("'") else " "
            return f"{new_subj} {n}{sep}{rest} pas"
    return f"ne {form} pas"


def generate_pairs(verbs: list[dict]) -> list[dict]:
    pairs = []
    seen = set()

    for v in verbs:
        inf = first_french_infinitive(v["fr"])
        if not inf:
            continue
        fr_forms = conjugate_french(inf)
        if not fr_forms:
            continue
        mp_stem = v["mp"]
        mp_forms = conjugate_mpongwe(mp_stem)

        for mood_fr in ("present", "future"):
            if mood_fr not in fr_forms or mood_fr not in mp_forms:
                continue
            fr_pos = fr_forms[mood_fr]
            fr_neg = [negate_french(f) for f in fr_pos]
            mp_pos = mp_forms[mood_fr]
            mp_neg = mp_forms.get(mood_fr + "_neg")
            for i in range(6):
                key = (mp_pos[i].lower(), fr_pos[i].lower())
                if key not in seen:
                    seen.add(key)
                    pairs.append({
                        "mpongwe": mp_pos[i],
                        "french": fr_pos[i],
                        "source": "aug_conjug",
                        "headword": mp_stem,
                        "gloss": inf,
                    })
                if mp_neg:
                    key = (mp_neg[i].lower(), fr_neg[i].lower())
                    if key not in seen:
                        seen.add(key)
                        pairs.append({
                            "mpongwe": mp_neg[i],
                            "french": fr_neg[i],
                            "source": "aug_conjug",
                            "headword": mp_stem,
                            "gloss": inf,
                        })

    return pairs


def main():
    print("=" * 60)
    print("Morphological Augmentation (verb conjugation)")
    print("=" * 60)

    data = json.loads(SOURCE_FILE.read_text(encoding="utf-8"))
    print(f"Dictionary entries: {len(data):,}")

    verbs = build_verb_list(data)
    print(f"Verb entries (v.*): {len(verbs):,}")

    # What's coverable?
    from collections import Counter
    cover = Counter()
    for v in verbs:
        inf = first_french_infinitive(v["fr"])
        if inf and conjugate_french(inf):
            cover["conjugable"] += 1
        else:
            cover["skipped"] += 1
    print(f"Conjugable: {cover['conjugable']:,} | Skipped (unsafe): {cover['skipped']:,}")

    pairs = generate_pairs(verbs)
    print(f"Generated pairs: {len(pairs):,}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Output: {OUTPUT_FILE}")

    print("\nSamples:")
    for p in pairs[:10]:
        print(f"  MP: {p['mpongwe']}")
        print(f"  FR: {p['french']}")
        print()


if __name__ == "__main__":
    main()