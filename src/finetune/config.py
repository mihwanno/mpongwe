from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILE = DATA_DIR / "mpongwe_francais.json"
GRAMMAR_FILE = DATA_DIR / "grammar.md"

OUTPUT_TRAIN = OUTPUT_DIR / "train.jsonl"
OUTPUT_VAL = OUTPUT_DIR / "val.jsonl"
OUTPUT_TEST = OUTPUT_DIR / "test.jsonl"
OUTPUT_GRAMMAR = OUTPUT_DIR / "grammar_pairs.jsonl"
OUTPUT_AUGMENTED = OUTPUT_DIR / "augmented_pairs.jsonl"

MPO_LANGUAGE_CODE = "mpo_Latn"
FRENCH_CODE = "fra_Latn"
ENGLISH_CODE = "eng_Latn"

VAL_SPLIT = 0.05
TEST_SPLIT = 0.05
MIN_PAIR_LENGTH = 2
MAX_PAIR_LENGTH = 100
