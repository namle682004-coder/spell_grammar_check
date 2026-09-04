
#!/usr/bin/env python
from src.inference import GrammarCorrector
from src.config import load_config
import argparse
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def split_sentences(text: str) -> list:
    """Split text into sentences."""
    # Split by Vietnamese/English sentence boundaries
    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÊẾỀỂỄỆÔỐỒỔỖỘƠỚỜỞỠỢƯỨỪỬỮỰA-Za-z])', text)
    return [s.strip() for s in sentences if s.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--text", "-t", required=True)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    corrector = GrammarCorrector(args.config)

    # Split into sentences
    sentences = split_sentences(args.text)

    if not args.quiet:
        print(f"\nFound {len(sentences)} sentences\n")
        print("-"*50)

    # Correct each sentence
    corrected_sentences = []
    for i, sent in enumerate(sentences):
        if not args.quiet:
            print(f"[{i+1}] Original: {sent}")

        result = corrector.correct_single(sent)

        # Extract only first line (corrected sentence)
        first_line = result.split('\n')[0].strip()
        corrected_sentences.append(first_line)

        if not args.quiet:
            print(f"    Corrected: {first_line}\n")

    # Combine
    full_text = ' '.join(corrected_sentences)

    print("="*50)
    print("FULL CORRECTED TEXT:")
    print("="*50)
    print(full_text)


if __name__ == "__main__":
    main()
