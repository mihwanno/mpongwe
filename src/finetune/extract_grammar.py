"""
Extract Mpongwe-French translation pairs from grammar.md.

Two phases:
1. Extract direct example sentences (Ex.: Mpongwe, French)
2. Generate synthetic pairs from grammar tables (noun classes, pronouns,
   verb conjugations, numbers, adjective agreement)
"""

import json
import re
from pathlib import Path

from config import GRAMMAR_FILE, OUTPUT_DIR

OUTPUT_FILE = OUTPUT_DIR / "grammar_pairs.jsonl"


# ---------------------------------------------------------------------------
# Phase 1: Extract direct example sentences
# ---------------------------------------------------------------------------

_EXPLANATION_MARKERS = (" (« ", " (litt. ", "litt.", "de para", ", de ",
                        "de dyena", "de kangana", "ou pour", " ; ")


def _clean_french(fr: str) -> str:
    """Remove explanatory parentheticals and truncate at Explanation markers."""
    fr = fr.strip()
    for marker in _EXPLANATION_MARKERS:
        idx = fr.find(marker)
        if idx != -1:
            fr = fr[:idx].rstrip(" ,;.")
    # Drop dangling explanation leftovers
    fr = fr.replace("« ", "").replace(" »", "").strip()
    return fr.rstrip(" ,;.")


def _add_if_clean(pairs: list[dict], seen: set, mp: str, fr: str) -> None:
    mp = re.sub(r"\*\*", "", mp).strip().strip("«»» “”")
    fr = _clean_french(re.sub(r"\*\*", "", fr))
    if mp == "Ex." or len(mp) < 2 or len(fr) < 2:
        return
    if any(c in fr for c in "«»"):
        return
    if re.search(r"(de|ou) [a-zà-ÿ\u0259œ]{2,}", fr):  # leftover explanation clause
        return
    pair = (mp.lower(), fr.lower())
    if pair in seen:
        return
    seen.add(pair)
    pairs.append({"mpongwe": mp, "french": fr, "source": "grammar_example"})


def extract_direct_examples() -> list[dict]:
    """Parse grammar.md for lines matching 'Ex. : Mpongwe, French' patterns."""
    text = GRAMMAR_FILE.read_text(encoding="utf-8")
    pairs = []
    seen = set()

    # Pattern 1: Ex. : **Mpongwe**, French  (bold Mpongwe)
    for m in re.finditer(
        r"Ex\.\s*:\s*\*\*(.+?)\*\*\s*,\s*(.+?)(?:\s*[;\n]|$)", text
    ):
        _add_if_clean(pairs, seen, m.group(1), m.group(2))

    # Pattern 2: Ex. : Mpongwe (no bold), French
    for m in re.finditer(
        r"Ex\.\s*:\s*([A-Z][a-zʼ\u02bcọẹọ̧̃̀̃\u0300-\u036f][^\n*;]{1,80}?)\s*,\s*([A-Z][^\n;]{1,120}?)(?:\s*[;\n]|$)",
        text,
    ):
        _add_if_clean(pairs, seen, m.group(1), m.group(2))

    return pairs


# ---------------------------------------------------------------------------
# Phase 2: Generate synthetic pairs from grammar tables
# ---------------------------------------------------------------------------

# Structured data extracted from grammar pages

NOUN_CLASSES = {
    "I":   {"name": "Personnel",   "sg_prefix": "o",  "pl_prefix": "a",  "type_sg": "oga, chef",         "type_pl": "aga",          "pron_sg": "w-", "pron_pl": "w-"},
    "II":  {"name": "Spécificatif","sg_prefix": "o",  "pl_prefix": "i",  "type_sg": "obo, barre de chaleur","type_pl": "ibo",       "pron_sg": "w-", "pron_pl": "y-"},
    "III": {"name": "Abstrait",    "sg_prefix": "o",  "pl_prefix": "a",  "type_sg": "ovono, ruse",        "type_pl": "avono",       "pron_sg": "w-", "pron_pl": "m-"},
    "IV":  {"name": "Extractif",   "sg_prefix": "o",  "pl_prefix": "i",  "type_sg": "ote, latte de raphia","type_pl": "ite",        "pron_sg": "w-", "pron_pl": "s-"},
    "V":   {"name": "Commun",      "sg_prefix": "i",  "pl_prefix": "i",  "type_sg": "nte, terre",         "type_pl": "inte",        "pron_sg": "y-", "pron_pl": "s-"},
    "VI":  {"name": "Noble",       "sg_prefix": "i",  "pl_prefix": "a",  "type_sg": "iba, mangue",        "type_pl": "aba",         "pron_sg": "b-", "pron_pl": "m-"},
    "VII": {"name": "Modal",       "sg_prefix": "e",  "pl_prefix": "ya", "type_sg": "eza, chose",         "type_pl": "ya",          "pron_sg": "z-", "pron_pl": "y-"},
}

NOUNS = [
    # (mpongwe_sg, french_sg, mpongwe_pl, french_pl, class_id, french_gender)
    ("oga", "le chef", "aga", "les chefs", "I", "m"),
    ("othwana", "l'enfant", "athwana", "les enfants", "I", "m"),
    ("obo", "la barre de chaleur", "ibo", "les barres de chaleur", "II", "f"),
    ("ovono", "la ruse", "avono", "les ruses", "III", "f"),
    ("ote", "la latte de raphia", "ite", "les lattes de raphia", "IV", "f"),
    ("nte", "la terre", "inte", "les terres", "V", "f"),
    ("iba", "la mangue", "aba", "les mangues", "VI", "f"),
    ("eza", "la chose", "ya", "les choses", "VII", "f"),
    ("mpoo", "le chemin", "impo", "les chemins", "IV", "m"),
    ("ohoro", "la bague", "ihoro", "les bagues", "II", "f"),
    ("ndjali", "le fusil", "intjali", "les fusils", "V", "m"),
    ("nango", "le remède", "iango", "les remèdes", "VII", "m"),
    ("osaka", "l'esclave", "asaka", "les esclaves", "I", "m"),
    ("onero", "le vieillard", "anero", "les vieillards", "I", "m"),
    ("egara", "la caisse", "agara", "les caisses", "VII", "f"),
    ("ngo", "l'habit", "ingo", "les habits", "VII", "m"),
    ("ehenoo", "la vie", "ahenoo", "les vies", "VII", "f"),
    ("ogumba", "le manioc", "agumba", "les maniocs", "I", "m"),
    ("nago", "la case", "agano", "les cases", "I", "f"),
    ("otega", "la barrique", "itega", "les barriques", "IV", "f"),
    ("igwera", "l'heure", "agwera", "les heures", "VI", "f"),
    ("ogwera", "la nuit", "agwera", "les nuits", "III", "f"),
    ("ntegu", "le jour", "itegu", "les jours", "V", "m"),
    ("mbute", "la bouteille", "ibute", "les bouteilles", "V", "f"),
    ("sika", "l'argent", None, None, "V", "m"),
    ("kyapu", "la capsule", None, None, "VII", "f"),
    ("ife", "la fièvre", None, None, "V", "f"),
    ("nare", "le bœuf", "agare", "les bœufs", "I", "m"),
    ("inigo", "l'eau", None, None, "III", "f"),
]

ADJECTIVES = [
    # (mpongwe_base, class_forms singular dict, class_forms plural dict)
    ("ombya", {
        "I": "ombya", "II": "ombya", "III": "ombya", "IV": "ombya",
        "V": "mbya", "VI": "lwya", "VII": "ewya",
    }, {
        "I": "awya", "II": "imbya", "III": "ambya", "IV": "ibya",
        "V": "imbya", "VI": "ambya", "VII": "wya",
    }),
    ("ompolo", {
        "I": "ompolo", "II": "mpolo", "III": "ivolo", "IV": "evolo",
        "V": "mpolo", "VI": "lwolo", "VII": "evolo",
    }, {
        "I": "awolo", "II": "impolo", "III": "ampolo", "IV": "ipolo",
        "V": "impolo", "VI": "ampolo", "VII": "volo",
    }),
    ("omwango", {
        "I": "omwango", "II": "nango", "III": "i\u1e47ango", "IV": "ezango",
        "V": "mpango", "VI": "liango", "VII": "ezango",
    }, {
        "I": "awango", "II": "imyango", "III": "athango", "IV": "idyango",
        "V": "ipango", "VI": "ahango", "VII": "yango",
    }),
]

# French adjective agreement forms: base -> {gender: {number: form}}
# BAGS adjectives that precede the noun in French.
FR_ADJECTIVES = {
    "ombya": "bon",
    "ompolo": "grand",
    "omwango": "petit",
}

def fr_adj_form(base: str, gender: str, plural: bool) -> str:
    """Return the French adjective form agreeing in gender and number."""
    irregular = {
        ("bon", "f", True): "bonnes",
        ("bon", "f", False): "bonne",
        ("bon", "m", True): "bons",
        ("bon", "m", False): "bon",
        ("grand", "f", True): "grandes",
        ("grand", "f", False): "grande",
        ("grand", "m", True): "grands",
        ("grand", "m", False): "grand",
        ("petit", "f", True): "petites",
        ("petit", "f", False): "petite",
        ("petit", "m", True): "petits",
        ("petit", "m", False): "petit",
    }
    return irregular.get((base, gender, plural), base)

PRONOUNS_PERSONAL = [
    # (mpongwe_subject, french_translation)
    ("mi", "je"),
    ("o", "tu"),
    ("e", "il"),
    ("azwe", "nous"),
    ("anwe", "vous"),
    ("wi", "ils"),
]

# Possessive adjective forms by noun class.
# Structure: {class_id: {"sg": {"mon": ..., "ton": ..., "son": ..., "notre": ..., "votre": ..., "leur": ...},
#                        "pl": {"mes": ..., "tes": ..., "ses": ..., "nos": ..., "vos": ..., "leurs": ...}}}
# From grammar page 647.
POSSESSIVE = {
    "I": {"sg": {"mon": "wani", "ton": "wo", "son": "we", "notre": "wazo", "votre": "wani", "leur": "wao"},
          "pl": {"mes": "wani", "tes": "wo", "ses": "we", "nos": "wazo", "vos": "wani", "leurs": "wao"}},
    "II": {"sg": {"mon": "wani", "ton": "wo", "son": "we", "notre": "wazo", "votre": "wani", "leur": "wao"},
           "pl": {"mes": "yani", "tes": "yo", "ses": "ye", "nos": "yazo", "vos": "yani", "leurs": "yao"}},
    "III": {"sg": {"mon": "wani", "ton": "wo", "son": "we", "notre": "wazo", "votre": "wani", "leur": "wao"},
            "pl": {"mes": "mani", "tes": "mo", "ses": "me", "nos": "mazo", "vos": "mani", "leurs": "mao"}},
    "IV": {"sg": {"mon": "wani", "ton": "wo", "son": "we", "notre": "wazo", "votre": "wani", "leur": "wao"},
           "pl": {"mes": "sani", "tes": "so", "ses": "se", "nos": "sazo", "vos": "sani", "leurs": "sao"}},
    "V": {"sg": {"mon": "yani", "ton": "yo", "son": "ye", "notre": "yazo", "votre": "yani", "leur": "yao"},
          "pl": {"mes": "sani", "tes": "so", "ses": "se", "nos": "sazo", "vos": "sani", "leurs": "sao"}},
    "VI": {"sg": {"mon": "pani", "ton": "go", "son": "pe", "notre": "pazo", "votre": "pani", "leur": "pao"},
           "pl": {"mes": "mani", "tes": "mo", "ses": "me", "nos": "mazo", "vos": "mani", "leurs": "mao"}},
    "VII": {"sg": {"mon": "zani", "ton": "zo", "son": "ze", "notre": "zazo", "votre": "zani", "leur": "zao"},
            "pl": {"mes": "yani", "tes": "yo", "ses": "ye", "nos": "yazo", "vos": "yani", "leurs": "yao"}},
}

# French possessive adjectives: gender -> {number: [forms]}. Order matches
# POSSESSIVE rows: mon/ton/son/notre/votre/leur (sg), mes/tes/ses/nos/vos/leurs (pl).
FR_POSS = {
    "m": {"sg": ["mon", "ton", "son", "notre", "votre", "leur"],
          "pl": ["mes", "tes", "ses", "nos", "vos", "leurs"]},
    "f": {"sg": ["ma", "ta", "sa", "notre", "votre", "leur"],
          "pl": ["mes", "tes", "ses", "nos", "vos", "leurs"]},
}

NUMBERS = [
    ("mori", "un"),
    ("mbani", "deux"),
    ("ntearo", "trois"),
    ("nai", "quatre"),
    ("nteani", "cinq"),
    ("orona", "six"),
    ("orowagenomu", "sept"),
    ("enani", "huit"),
    ("endoghi", "neuf"),
    ("igothi", "dix"),
    ("nkama", "cent"),
    ("ntozeni", "mille"),
]

# Number agreement forms per noun class (grammar page 645).
# Singular = for "un/une" (1); plural = for deux/trois/quatre/cinq (2-5).
NUMBERS_BY_CLASS = {
    "I":   {"1": "othori", "2": "awani", "3": "araro", "4": "anai", "5": "atani"},
    "II":  {"1": "othori", "2": "imbani", "3": "iraro", "4": "inai", "5": "itani"},
    "III": {"1": "othori", "2": "ambani", "3": "araro", "4": "anai", "5": "atani"},
    "IV":  {"1": "othori", "2": "bani", "3": "taro", "4": "nai", "5": "tani"},
    "V":   {"1": "mori", "2": "mbani", "3": "ntearo", "4": "nai", "5": "nteani"},
    "VI":  {"1": "imbori", "2": "ambani", "3": "araro", "4": "anai", "5": "atani"},
    "VII": {"1": "enhori", "2": "wani", "3": "raro", "4": "nai", "5": "tani"},
}
NUMBERS_FR = {"1": "un", "2": "deux", "3": "trois", "4": "quatre", "5": "cinq"}

# Full verb conjugation: KAMBA (parler)
VERB_KAMBA = {
    "infinitive": ("kamba", "parler"),
    "imperative": ("gamba", "parle"),
    "imperative_pl": ("gambani", "parlez"),
    "indicative_present": [
        ("mi kamba", "je parle"),
        ("o kamba", "tu parles"),
        ("e kamba", "il parle"),
        ("azwe kamba", "nous parlons"),
        ("anwe kamba", "vous parlez"),
        ("wi kamba", "ils parlent"),
    ],
    "indicative_present_neg": [
        ("mi pa kamba", "je ne parle pas"),
        ("o pa kamba", "tu ne parles pas"),
        ("e pa kamba", "il ne parle pas"),
        ("azwe pa kamba", "nous ne parlons pas"),
        ("anwe pa kamba", "vous ne parlez pas"),
        ("wi pa kamba", "ils ne parlent pas"),
    ],
    "imparfait_immediat": [
        ("myakambaga", "je parlais"),
        ("o'akambaga", "tu parlais"),
        ("akambaga", "il parlait"),
        ("azw'akambaga", "nous parlions"),
        ("anw'akambaga", "vous parliez"),
        ("w'akambaga", "ils parlaient"),
    ],
    "passe_simple": [
        ("myakambi", "j'ai parlé"),
        ("o'akambi", "tu as parlé"),
        ("akambi", "il a parlé"),
        ("azw'akambi", "nous avons parlé"),
        ("anw'akambi", "vous avez parlé"),
        ("w'akambi", "ils ont parlé"),
    ],
    "futur": [
        ("mi be kamba", "je parlerai"),
        ("o be kamba", "tu parleras"),
        ("e be kamba", "il parlera"),
        ("azwe be kamba", "nous parlerons"),
        ("anwe be kamba", "vous parlerez"),
        ("wi be kamba", "ils parleront"),
    ],
    "futur_neg": [
        ("m'be kamba", "je ne parlerai pas"),
        ("o'be kamba", "tu ne parleras pas"),
        ("e'be kamba", "il ne parlera pas"),
        ("azwe'be kamba", "nous ne parlerons pas"),
        ("anwe'be kamba", "vous ne parlerez pas"),
        ("wi'be kamba", "ils ne parleront pas"),
    ],
    "conditionnel": [
        ("mi to kamba", "si je parle"),
        ("o to kamba", "si tu parles"),
        ("e to kamba", "s'il parle"),
        ("azwe to kamba", "si nous parlons"),
        ("anwe to kamba", "si vous parlez"),
        ("wi to kamba", "s'ils parlent"),
    ],
    "subjonctif": [
        ("mi ga gambe", "que je parle"),
        ("gamba", "que tu parles"),
        ("e ga gambe", "qu'il parle"),
        ("azwe ga gambe", "que nous parlions"),
        ("anwe ga gambe", "que vous parliez"),
        ("wi ga gambe", "qu'ils parlent"),
    ],
}

# Full verb conjugation: RE (être)
VERB_RE = {
    "infinitive": ("re", "être"),
    "present": [
        ("myare", "je suis"),
        ("o re", "tu es"),
        ("are", "il est"),
        ("azware", "nous sommes"),
        ("anware", "vous êtes"),
        ("wi re", "ils sont"),
    ],
    "present_neg": [
        ("myazele", "je ne suis pas"),
        ("o zele", "tu n'es pas"),
        ("azele", "il n'est pas"),
        ("azwazele", "nous ne sommes pas"),
        ("anwazele", "vous n'êtes pas"),
        ("wi zele", "ils ne sont pas"),
    ],
    "imparfait_imm_1": [
        ("myapegaga", "j'étais"),
        ("o pegaga", "tu étais"),
        ("apegaga", "il était"),
        ("azwapegaga", "nous étions"),
        ("anwapegaga", "vous étiez"),
        ("wapegaga", "ils étaient"),
    ],
    "imparfait_imm_2": [
        ("myaduo", "j'étais"),
        ("o duo", "tu étais"),
        ("aduo", "il était"),
        ("azwaduo", "nous étions"),
        ("anwaduo", "vous étiez"),
        ("waduo", "ils étaient"),
    ],
    "imparfait_eloigne": [
        ("myavegagi", "j'étais"),
        ("ovegagi", "tu étais"),
        ("avegagi", "il était"),
        ("azwavegagi", "nous étions"),
        ("anwavegagi", "vous étiez"),
        ("wavegagi", "ils étaient"),
    ],
    "futur": [
        ("mi be duo", "je serai"),
        ("o be duo", "tu seras"),
        ("e be duo", "il sera"),
        ("azwe be duo", "nous serons"),
        ("anwe be duo", "vous serez"),
        ("wi be duo", "ils seront"),
    ],
    "futur_neg": [
        ("m'be duo", "je ne serai pas"),
        ("o'be duo", "tu ne seras pas"),
        ("e'be duo", "il ne sera pas"),
        ("azwe'be duo", "nous ne serons pas"),
        ("anwe'be duo", "vous ne serez pas"),
        ("wi'be duo", "ils ne seront pas"),
    ],
    "conditionnel": [
        ("mi to duo", "si je suis"),
        ("o to duo", "si tu es"),
        ("e to duo", "s'il est"),
        ("azwe to duo", "si nous sommes"),
        ("anwe to duo", "si vous êtes"),
        ("wi to duo", "s'ils sont"),
    ],
}


def generate_verb_pairs() -> list[dict]:
    """Generate all conjugated verb pairs."""
    pairs = []
    seen = set()

    for verb_table in [VERB_KAMBA, VERB_RE]:
        for tense, forms in verb_table.items():
            if tense == "infinitive":
                mp, fr = forms
                key = (mp, fr)
                if key not in seen:
                    seen.add(key)
                    pairs.append({"mpongwe": mp, "french": fr, "source": "grammar_verb"})
                continue
            if not isinstance(forms, list):
                continue
            for mp, fr in forms:
                key = (mp.lower(), fr.lower())
                if key not in seen:
                    seen.add(key)
                    pairs.append({"mpongwe": mp, "french": fr, "source": "grammar_verb"})

    # Derived verb forms from grammar
    derived = [
        ("sayona", "se tuer"),
        ("sovvuma", "se baigner"),
        ("adyon'oku'weme", "il s'est tué"),
        ("mife kenda", "je repars"),
        ("no tiga", "cesse d'abord"),
        ("no tigare", "attends un peu"),
        ("no na", "mange d'abord"),
        ("no nare", "mange d'abord"),
        ("dyena", "voir"),
        ("dyeno", "être vu"),
        ("bolawola", "battre souvent"),
        ("bolawolo", "être battu souvent"),
        ("myayeni", "j'ai vu"),
        ("myayeno", "j'ai été vu"),
        ("wi ga veye", "qu'ils appellent"),
        ("wi ga veyo", "qu'ils soient appelés"),
        ("dyenaga", "voir fréquemment"),
        ("pyagaga", "brûler continuellement"),
        ("dyenago", "être vu fréquemment"),
        ("denalena", "pleurer continuellement"),
        ("dena", "pleurer"),
        ("todawora", "insulter beaucoup"),
        ("towa", "insulter"),
        ("denalenaga", "pleurer constamment"),
        ("dyenana", "se voir"),
        ("nungwana", "s'entr'aider"),
        ("myagana", "se connaître"),
        ("mya", "connaître"),
        ("tendina oma", "écrire à quelqu'un"),
        ("tenda", "écrire"),
        ("bendina oma", "se fâcher contre quelqu'un"),
        ("benda", "se fâcher"),
        ("byena oma", "venir à quelqu'un"),
        ("bya", "venir"),
        ("abendino", "on s'est fâché contre lui"),
        ("dyogino", "obéir"),
        ("dyogo", "entendre"),
        ("tonday", "faire aimer"),
        ("tonda", "aimer"),
        ("poswa", "tomber"),
        ("posunya", "faire tomber"),
        ("dyandjiza", "faire travailler"),
        ("dyandja", "travailler"),
        ("byeza", "faire venir"),
        ("paruna", "enlever des mains"),
        ("para", "porter dans les mains"),
        ("taluna", "retirer, déblayer"),
        ("talya", "superposer, remblayer"),
        ("mèna", "avaler"),
        ("nana", "dormir"),
        ("na", "manger"),
        ("bena", "planter"),
        ("denda", "faire"),
        ("feya", "appeler"),
        ("kenda", "aller"),
        ("saza", "effacer"),
        ("tanga", "compter"),
        ("baga", "apporter"),
        ("waga", "apporte"),
        ("yena", "vois"),
    ]

    for mp, fr in derived:
        key = (mp.lower(), fr.lower())
        if key not in seen:
            seen.add(key)
            pairs.append({"mpongwe": mp, "french": fr, "source": "grammar_verb"})

    return pairs


def generate_noun_class_pairs() -> list[dict]:
    """Generate singular/plural noun pairs from noun class data."""
    pairs = []
    seen = set()

    for mp_sg, fr_sg, mp_pl, fr_pl, cls, gender in NOUNS:
        if mp_pl and fr_pl:
            key = (mp_sg.lower(), fr_sg.lower())
            if key not in seen:
                seen.add(key)
                pairs.append({"mpongwe": mp_sg, "french": fr_sg, "source": "grammar_noun"})
            key = (mp_pl.lower(), fr_pl.lower())
            if key not in seen:
                seen.add(key)
                pairs.append({"mpongwe": mp_pl, "french": fr_pl, "source": "grammar_noun"})

    return pairs


def strip_article(fr: str) -> str:
    """Remove leading article (le/la/les/l'/du/de la/des) from French noun."""
    m = re.match(r"^(les|des|de la|le|la|l'|d'|du)\s*(.+)$", fr)
    if m:
        return m.group(2)
    return fr


def generate_pronoun_noun_pairs() -> list[dict]:
    """Generate pronoun + noun combinations (possessive + noun)."""
    pairs = []
    seen = set()

    for mp_sg, fr_sg, mp_pl, fr_pl, cls, gender in NOUNS:
        base_sg = strip_article(fr_sg)
        base_pl = strip_article(fr_pl) if fr_pl else None
        poss = POSSESSIVE.get(cls, {})
        fr_forms = FR_POSS.get(gender, FR_POSS["m"])

        # Singular noun possessive: "oga wani" = "mon chef"
        # Mpongwe possessive rows: mon/ton/son/notre/votre/leur
        if mp_sg and base_sg:
            for i, mp_form_name in enumerate(["mon", "ton", "son", "notre", "votre", "leur"]):
                mp_form = poss.get("sg", {}).get(mp_form_name)
                fr_adj = fr_forms["sg"][i]
                if mp_form:
                    combo = f"{mp_sg} {mp_form}"
                    fr_combo = f"{fr_adj} {base_sg}"
                    key = (combo.lower(), fr_combo.lower())
                    if key not in seen:
                        seen.add(key)
                        pairs.append({"mpongwe": combo, "french": fr_combo, "source": "grammar_possessive"})

        # Plural noun possessive: "aga wani" = "mes chefs"
        if mp_pl and base_pl:
            for i, mp_form_name in enumerate(["mes", "tes", "ses", "nos", "vos", "leurs"]):
                mp_form = poss.get("pl", {}).get(mp_form_name)
                fr_adj = fr_forms["pl"][i]
                if mp_form:
                    combo = f"{mp_pl} {mp_form}"
                    fr_combo = f"{fr_adj} {base_pl}"
                    key = (combo.lower(), fr_combo.lower())
                    if key not in seen:
                        seen.add(key)
                        pairs.append({"mpongwe": combo, "french": fr_combo, "source": "grammar_possessive"})

    return pairs


def generate_number_noun_pairs() -> list[dict]:
    """Generate number + noun combinations with per-class agreement."""
    pairs = []
    seen = set()

    for mp_sg, fr_sg, mp_pl, fr_pl, cls, gender in NOUNS:
        class_nums = NUMBERS_BY_CLASS.get(cls, {})
        base_pl = strip_article(fr_pl) if fr_pl else None

        # Numbers 2-5 with plural nouns: "awani aga" = "deux chefs"
        if mp_pl and base_pl:
            for num_key in ["2", "3", "4", "5"]:
                num_mp = class_nums.get(num_key)
                num_fr = NUMBERS_FR.get(num_key)
                base_noun_pl = strip_article(fr_pl)
                combo = f"{num_mp} {mp_pl}"
                combo_fr = f"{num_fr} {base_noun_pl}"
                key = (combo.lower(), combo_fr.lower())
                if key not in seen:
                    seen.add(key)
                    pairs.append({"mpongwe": combo, "french": combo_fr, "source": "grammar_number"})

        # Number 1 with singular noun: "mori ote" = "une latte"
        num_mp = class_nums.get("1")
        base_sg = strip_article(fr_sg)
        if num_mp and mp_sg and base_sg:
            article = "un" if gender == "m" else "une"
            combo = f"{num_mp} {mp_sg}"
            combo_fr = f"{article} {base_sg}"
            key = (combo.lower(), combo_fr.lower())
            if key not in seen:
                seen.add(key)
                pairs.append({"mpongwe": combo, "french": combo_fr, "source": "grammar_number"})

    # Cardinal number pairs
    for mp, fr in NUMBERS:
        key = (mp.lower(), fr.lower())
        if key not in seen:
            seen.add(key)
            pairs.append({"mpongwe": mp, "french": fr, "source": "grammar_number"})

    return pairs


def generate_adjective_noun_pairs() -> list[dict]:
    """Generate adjective + noun agreement pairs (BAGS adjectives precede noun in French)."""
    pairs = []
    seen = set()

    for adj_base, sg_forms, pl_forms in ADJECTIVES:
        fr_base = FR_ADJECTIVES[adj_base]
        for mp_sg, fr_sg, mp_pl, fr_pl, cls, gender in NOUNS:
            base_sg = strip_article(fr_sg)
            base_pl = strip_article(fr_pl) if fr_pl else None

            # Singular: "un bon chef" / "une bonne chose"
            sg_form = sg_forms.get(cls)
            if sg_form and mp_sg:
                combo = f"{mp_sg} {sg_form}"
                article = "un" if gender == "m" else "une"
                fr_adj = fr_adj_form(fr_base, gender, plural=False)
                fr_combo = f"{article} {fr_adj} {base_sg}"
                key = (combo.lower(), fr_combo.lower())
                if key not in seen:
                    seen.add(key)
                    pairs.append({"mpongwe": combo, "french": fr_combo, "source": "grammar_adjective"})

            # Plural: "des bons chefs" / "des bonnes choses"
            pl_form = pl_forms.get(cls)
            if pl_form and mp_pl and base_pl:
                combo = f"{mp_pl} {pl_form}"
                fr_adj = fr_adj_form(fr_base, gender, plural=True)
                fr_combo = f"des {fr_adj} {base_pl}"
                key = (combo.lower(), fr_combo.lower())
                if key not in seen:
                    seen.add(key)
                    pairs.append({"mpongwe": combo, "french": fr_combo, "source": "grammar_adjective"})

    return pairs


# ---------------------------------------------------------------------------
# Phase 3: Additional grammar-rule based sentences
# ---------------------------------------------------------------------------

def generate_grammar_rule_sentences() -> list[dict]:
    """
    Generate sentences from explicit grammar rule examples.
    These are the full sentences from grammar.md that explain patterns.
    """
    sentences = [
        # Noun classes
        ("othwana", "un enfant"),
        ("othwana onothe", "un garçon"),
        ("othwana othwanto", "une fille"),
        ("nare nome", "un taureau"),
        ("nare ganton", "une vache"),
        ("oma", "quelqu'un, une personne"),
        ("mongi", "des gens"),
        ("aboswe", "ancêtres"),
        ("aninigo", "eau"),

        # Genitive construction (de)
        ("mpoo ya gare", "le chemin du milieu"),
        ("impo sa gare", "les chemins du milieu"),
        ("ebanda zi nanha", "la peau de l'animal"),
        ("banda iyanha", "les peaux des animaux"),
        ("oga w'agekaza", "le chef des Agékaza"),
        ("aga w'agekaza", "les chefs des Agékaza"),

        # Preposition en/a/pour
        ("ohoro wi sika", "une bague en argent"),
        ("ndjali yi kyapu", "un fusil à capsule"),
        ("nango y'ife", "un remède pour la fièvre"),

        # Adjective agreement
        ("ogara ralye", "une caisse pleine"),
        ("ngo ya pupu", "un habit blanc"),
        ("ehenoo za pekepeke", "la vie éternelle"),
        ("owaro owya", "une belle pirogue"),
        ("orerembo ovolo", "la grande rivière"),

        # Noun as adjective
        ("osakaw'ekale", "un esclave orgueilleux"),
        ("asaka wi kale", "des esclaves orgueilleux"),
        ("onero w'ipoku", "un vieillard aveugle"),
        ("anero w'apoku", "des vieillards aveugles"),
        ("aningo mi mpyo", "de l'eau chaude"),
        ("imongo s'onigi", "des patates douces"),

        # Attribute with re/twa
        ("egara zi re ralye", "la caisse est pleine"),
        ("aningo m'atweni mpyo", "l'eau est chaude"),
        ("asaroe windé wi re apoku", "ces vieillards sont aveugles"),
        ("asaka w'atweni kale", "les esclaves sont devenus orgueilleux"),

        # Ordinal numbers
        ("oga wi mbani", "le deuxième roi"),
        ("ivanga ya mbani", "le deuxième commandement"),
        ("ntegu yi ntaro", "le troisième jour"),
        ("owaro wo raro", "la troisième pirogue"),
        ("otenbo wi naf", "le quatrième doigt"),
        ("mbwa yi ntani", "le cinquième chien"),
        ("othwana wa tani", "le cinquième enfant"),
        ("oma wa nai", "la quatrième personne"),

        # Fractions
        ("erene z'ogumba", "la moitié d'un manioc"),
        ("mbet yi nago", "la moitié d'une case"),
        ("otene w'otega", "la moitié d'une barrique"),
        ("erene z'igwera", "une demi-heure"),

        # Possessive pronoun
        ("epele zañi", "mon assiette"),
        ("epele zino za mande", "À qui est cette assiette"),
        ("izamhi", "c'est la mienne"),
        ("og'iwazo", "notre chef"),
        ("swak'iyamhi", "mon couteau"),
        ("ab'imao", "leurs mangues"),

        # Demonstrative
        ("ogo winó", "ce bras"),
        ("wino e dyona", "celui-ci rit"),
        ("wono e dena", "celui-là pleure"),
        ("yino mpono mbe", "ce chemin est mauvais"),

        # Verb être examples
        ("mi are", "je suis"),
        ("o re", "tu es"),
        ("are", "il est"),
        ("azw'are", "nous sommes"),
        ("anw'are", "vous êtes"),
        ("wi re", "ils sont"),
        ("mi azele", "je ne suis pas"),
        ("o zele", "tu n'es pas"),
        ("azele", "il n'est pas"),

        # Verb kamba examples
        ("mi kamba", "je parle"),
        ("o kamba", "tu parles"),
        ("e kamba", "il parle"),
        ("azwe kamba", "nous parlons"),
        ("anwe kamba", "vous parlez"),
        ("wi kamba", "ils parlent"),
        ("mi pa kamba", "je ne parle pas"),
        ("o pa kamba", "tu ne parles pas"),
        ("e pa kamba", "il ne parle pas"),

        # Auxiliary duo/aluo constructions
        ("myaduo mi kamba", "je parlais (autrefois)"),
        ("mi aluo mi kamba", "je parlais (hier)"),
        ("myaduo myakambi", "j'avais parlé (autrefois)"),
        ("mi aluo myakambi", "j'avais parlé (hier)"),
        ("mi be duo mi kamba", "je parlerai"),
        ("mi be duo myakambi", "j'aurai parlé"),
        ("mi duo mi kamba", "je ne parlais pas"),
        ("mi duo myakambi", "j'avais pas parlé"),
        ("mi be duo mi kamba", "je ne parlerai pas"),

        # Négation with tonal shift
        ("mi ato duo", "j'aurais été"),
        ("mi to duo", "je ne serais pas"),

        # Derived verb forms
        ("sayona", "se tuer"),
        ("sovuma", "se baigner"),
        ("adyon'oku'weme", "il s'est tué"),
        ("mife kenda", "je repars"),
        ("no tiga", "cesse d'abord"),
        ("dyena", "voir"),
        ("dyeno", "être vu"),
        ("dyenaga", "voir fréquemment"),
        ("dyenago", "être vu fréquemment"),
        ("dyenana", "se voir"),
        ("nungwana", "s'entr'aider"),
        ("myagana", "se connaître"),
        ("tendina oma", "écrire à quelqu'un"),
        ("bendina oma", "se fâcher contre quelqu'un"),
        ("byena oma", "venir à quelqu'un"),
        ("tonday", "faire aimer"),
        ("posunya", "faire tomber"),
        ("dyandjiza", "faire travailler"),
        ("byeza", "faire venir"),
        ("paruna", "enlever des mains"),
        ("taluna", "retirer, déblayer"),
        ("dyogino", "obéir"),

        # Imperative
        ("gamba", "parle"),
        ("gambani", "parlez"),
        ("agamba", "ne parle pas"),
        ("agambani", "ne parlez pas"),
        ("yena", "vois"),
        ("nana", "dors"),
        ("na", "mange"),
        ("bena", "plante"),
        ("denda", "fais"),
        ("kenda", "va"),
        ("saza", "efface"),
        ("tanga", "compte"),
        ("baga", "apporte"),
        ("waga", "apporte"),
        ("yogo", "viens"),

        # Infinitives
        ("kamba", "parler"),
        ("re", "être"),
        ("dyuwa", "mourir"),
        ("dyandja", "travailler"),
        ("dwana", "rester, demeurer"),
        ("bonga", "prendre"),
        ("tenda", "écrire"),
        ("benda", "se fâcher"),
        ("bya", "venir"),
        ("tonda", "aimer"),
        ("poswa", "tomber"),
        ("para", "porter dans les mains"),
        ("talya", "superposer"),
        ("mya", "connaître"),
        ("dena", "pleurer"),
        ("towa", "insulter"),

        # Subjonctif
        ("mi ga gambe", "que je parle"),
        ("e ga gambe", "qu'il parle"),
        ("azwe ga gambe", "que nous parlions"),
        ("anwe ga gambe", "que vous parliez"),
        ("wi ga gambe", "qu'ils parlent"),

        # Additional from dictionary page 640
        ("zoro", "soudain, à l'improviste"),
        ("zuguzugu", "encombré, bondé"),
        ("zuthine", "le crâne a entièrement baissé"),
        ("zunge", "fais vite"),
        ("zuzee", "étroit, resserré"),
        ("zwere", "très froid, glacé"),
        ("zwè", "nous"),
        ("vani zwè alasa araro", "donnez-nous trois oranges"),
        ("anambye e dyena zwè", "Dieu nous voit"),
        ("aranga zwè", "ce n'est pas nous"),
        ("aza zwè", "sans nous"),
        ("ga n'are zwè", "comme nous sommes"),
    ]

    pairs = []
    seen = set()
    for mp, fr in sentences:
        key = (mp.lower().strip(), fr.lower().strip())
        if key not in seen and len(mp) > 1 and len(fr) > 1:
            seen.add(key)
            pairs.append({"mpongwe": mp.strip(), "french": fr.strip(), "source": "grammar_rule"})

    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Grammar Extraction & Synthetic Generation")
    print("=" * 60)

    # Phase 1: Direct examples
    direct = extract_direct_examples()
    print(f"Phase 1 - Direct examples from grammar.md: {len(direct)} pairs")

    # Phase 2: Structured table generation
    verb_pairs = generate_verb_pairs()
    print(f"Phase 2a - Verb conjugation pairs: {len(verb_pairs)} pairs")

    noun_pairs = generate_noun_class_pairs()
    print(f"Phase 2b - Noun class pairs: {len(noun_pairs)} pairs")

    pronoun_pairs = generate_pronoun_noun_pairs()
    print(f"Phase 2c - Pronoun + noun pairs: {len(pronoun_pairs)} pairs")

    number_pairs = generate_number_noun_pairs()
    print(f"Phase 2d - Number + noun pairs: {len(number_pairs)} pairs")

    adj_pairs = generate_adjective_noun_pairs()
    print(f"Phase 2e - Adjective + noun pairs: {len(adj_pairs)} pairs")

    # Phase 3: Explicit grammar rule sentences
    rule_pairs = generate_grammar_rule_sentences()
    print(f"Phase 3 - Grammar rule sentences: {len(rule_pairs)} pairs")

    # Combine all
    all_pairs = direct + verb_pairs + noun_pairs + pronoun_pairs + number_pairs + adj_pairs + rule_pairs

    # Deduplicate
    seen = set()
    unique_pairs = []
    for p in all_pairs:
        key = (p["mpongwe"].lower().strip(), p["french"].lower().strip())
        if key not in seen and len(p["mpongwe"]) > 1 and len(p["french"]) > 1:
            seen.add(key)
            unique_pairs.append(p)

    print(f"\nTotal unique pairs: {len(unique_pairs)}")

    # Write output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in unique_pairs:
            record = {
                "translation": {
                    "mpo_Latn": pair["mpongwe"],
                    "fra_Latn": pair["french"],
                },
                "source": pair["source"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nOutput: {OUTPUT_FILE}")

    # Summary by source
    sources = {}
    for p in unique_pairs:
        sources[p["source"]] = sources.get(p["source"], 0) + 1
    print("\nBy source:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    # Show samples
    print("\nSamples:")
    import random
    random.seed(42)
    for p in random.sample(unique_pairs, min(10, len(unique_pairs))):
        print(f"  MP: {p['mpongwe']}")
        print(f"  FR: {p['french']}")
        print()


if __name__ == "__main__":
    main()
