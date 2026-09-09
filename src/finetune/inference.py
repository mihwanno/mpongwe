"""
Inference script for Mpongwe translation using fine-tuned NLLB model.

Usage:
    python inference.py --model_path ./output/final

    # Interactive mode
    python inference.py --model_path ./output/final --interactive
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from config import MPO_LANGUAGE_CODE, FRENCH_CODE, ENGLISH_CODE


class MpongweTranslator:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.mpo_id = self.tokenizer.convert_tokens_to_ids(MPO_LANGUAGE_CODE)
        self.fr_id = self.tokenizer.convert_tokens_to_ids(FRENCH_CODE)
        self.en_id = self.tokenizer.convert_tokens_to_ids(ENGLISH_CODE)

        print(f"Device: {self.device}")

    def translate(self, text: str, src_lang: str, tgt_lang: str, max_length: int = 256) -> str:
        self.tokenizer.src_lang = src_lang
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        tgt_id = self._get_lang_id(tgt_lang)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=tgt_id,
                max_length=max_length,
                num_beams=4,
                temperature=1.0,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _get_lang_id(self, lang: str) -> int:
        lang_ids = {
            MPO_LANGUAGE_CODE: self.mpo_id,
            FRENCH_CODE: self.fr_id,
            ENGLISH_CODE: self.en_id,
        }
        return lang_ids.get(lang, self.tokenizer.convert_tokens_to_ids(lang))
    
    def mpongwe_to_french(self, text: str) -> str:
        return self.translate(text, MPO_LANGUAGE_CODE, FRENCH_CODE)
    
    def french_to_mpongwe(self, text: str) -> str:
        return self.translate(text, FRENCH_CODE, MPO_LANGUAGE_CODE)
    
    def english_to_mpongwe(self, text: str) -> str:
        return self.translate(text, ENGLISH_CODE, MPO_LANGUAGE_CODE)
    
    def mpongwe_to_english(self, text: str) -> str:
        return self.translate(text, MPO_LANGUAGE_CODE, ENGLISH_CODE)


def interactive_mode(translator: MpongweTranslator):
    print("\n" + "=" * 60)
    print("Mpongwe Translation (Interactive Mode)")
    print("=" * 60)
    print("Commands:")
    print("  mp> <text>  - Mpongwe to French")
    print("  fr> <text>  - French to Mpongwe")
    print("  en> <text>  - English to Mpongwe")
    print("  quit        - Exit")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            
            if user_input.startswith("mp>"):
                text = user_input[3:].strip()
                result = translator.mpongwe_to_french(text)
                print(f"French: {result}")
            
            elif user_input.startswith("fr>"):
                text = user_input[3:].strip()
                result = translator.french_to_mpongwe(text)
                print(f"Mpongwe: {result}")
            
            elif user_input.startswith("en>"):
                text = user_input[3:].strip()
                result = translator.english_to_mpongwe(text)
                print(f"Mpongwe: {result}")
            
            else:
                print("Unknown command. Use mp>, fr>, or en>")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def demo_mode(translator: MpongweTranslator):
    print("\n" + "=" * 60)
    print("Demo Translations")
    print("=" * 60)
    
    test_cases = [
        ("Mpongwe → French", "mpongwe_to_french", "Pusi yi daga"),
        ("Mpongwe → French", "mpongwe_to_french", "Ntee yíno yi sulaga ni vunga"),
        ("French → Mpongwe", "french_to_mpongwe", "le chat miaule"),
        ("French → Mpongwe", "french_to_mpongwe", "il y a toujours des épidémies"),
    ]
    
    for desc, method, text in test_cases:
        func = getattr(translator, method)
        result = func(text)
        print(f"\n{desc}:")
        print(f"  Input: {text}")
        print(f"  Output: {result}")


def main():
    parser = argparse.ArgumentParser(description="Mpongwe Translation Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--text", type=str, help="Text to translate")
    parser.add_argument("--src", type=str, default="fr", choices=["mp", "fr", "en"], help="Source language")
    parser.add_argument("--tgt", type=str, default="mp", choices=["mp", "fr", "en"], help="Target language")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    args = parser.parse_args()
    
    translator = MpongweTranslator(args.model_path, device=args.device)
    
    if args.interactive:
        interactive_mode(translator)
    elif args.text:
        src_map = {"mp": MPO_LANGUAGE_CODE, "fr": FRENCH_CODE, "en": ENGLISH_CODE}
        result = translator.translate(args.text, src_map[args.src], src_map[args.tgt])
        print(result)
    else:
        demo_mode(translator)


if __name__ == "__main__":
    main()
