import argparse
import sys
from pathlib import Path

from src.config import load_config
from src.inference import GrammarCorrector
from src.utils import set_seed, setup_logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Interactive grammar correction")
    parser.add_argument("--config", "-c", required=True,
                        help="Path to config file")
    parser.add_argument(
        "--text", "-t", help="Text to correct (optional, if not provided will enter interactive mode)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup
    set_seed(config.get('seed', 42))
    logger = setup_logger(__name__)

    # Initialize corrector
    logger.info("Loading model...")
    corrector = GrammarCorrector(args.config)

    if args.text:
        # Single text mode
        print(f"\nInput: {args.text}")
        result = corrector.correct_single(args.text)
        print(f"Output:\n{result}")
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("Grammar Correction - Interactive Mode")
        print("Type 'exit' or 'quit' to stop")
        print("="*50 + "\n")

        while True:
            try:
                text = input("Enter text to correct: ").strip()

                if text.lower() in ['exit', 'quit', '']:
                    print("Goodbye!")
                    break

                print("\nProcessing...")
                result = corrector.correct_single(text)
                print("\n" + "-"*40)
                print(f"Original: {text}")
                print(f"Corrected:\n{result}")
                print("-"*40 + "\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
