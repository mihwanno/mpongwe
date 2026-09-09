"""
Add Mpongwe language token to NLLB tokenizer and save modified model.

NLLB-200 has fixed language tokens. We need to:
1. Add a new token 'mpo_Latn' to the tokenizer
2. Resize model embeddings to accommodate the new token
3. Save the modified model for fine-tuning
"""

import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from config import MPO_LANGUAGE_CODE, OUTPUT_DIR


def add_mpongwe_token(
    model_name: str = "facebook/nllb-200-distilled-600M",
    output_dir: Path = None,
):
    output_dir = output_dir or OUTPUT_DIR / "nllb-mpongwe-base"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print(f"Original vocab size: {len(tokenizer)}")
    print(f"Original model embeddings: {model.model.shared.num_embeddings}")
    
    existing_tokens = list(tokenizer.added_tokens_encoder.keys())
    similar_langs = [t for t in existing_tokens if "Latn" in t and t.startswith(("swa", "lin", "kin", "fra"))]
    print(f"\nSimilar Bantu language tokens: {similar_langs[:10]}")
    
    num_added = tokenizer.add_tokens([MPO_LANGUAGE_CODE])
    print(f"\nAdded {num_added} new token(s): {MPO_LANGUAGE_CODE}")
    
    model.resize_token_embeddings(len(tokenizer))
    print(f"New vocab size: {len(tokenizer)}")
    print(f"New model embeddings: {model.model.shared.num_embeddings}")
    
    new_token_id = tokenizer.convert_tokens_to_ids(MPO_LANGUAGE_CODE)
    print(f"\n'{MPO_LANGUAGE_CODE}' token ID: {new_token_id}")
    
    similar_token = "swa_Latn"
    similar_id = tokenizer.convert_tokens_to_ids(similar_token)
    
    with torch.no_grad():
        similar_embedding = model.model.shared.weight[similar_id].clone()
        model.model.shared.weight[new_token_id] = similar_embedding
        model.model.encoder.embed_tokens.weight[new_token_id] = similar_embedding
        model.model.decoder.embed_tokens.weight[new_token_id] = similar_embedding
    
    print(f"Initialized '{MPO_LANGUAGE_CODE}' embedding from '{similar_token}'")
    
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    
    print(f"\nSaved modified model to: {output_dir}")
    
    test_tokenizer = AutoTokenizer.from_pretrained(output_dir)
    test_id = test_tokenizer.convert_tokens_to_ids(MPO_LANGUAGE_CODE)
    print(f"Verification - '{MPO_LANGUAGE_CODE}' token ID: {test_id}")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add mpo_Latn token to NLLB")
    parser.add_argument("--model_name", type=str,
                        default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--output_dir", type=str,
                        default=str(OUTPUT_DIR / "nllb-mpongwe-base"))
    args = parser.parse_args()
    add_mpongwe_token(args.model_name, Path(args.output_dir))
