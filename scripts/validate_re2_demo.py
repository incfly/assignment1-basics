from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from re2_demo import findall


def main() -> None:
    examples = [
        ("hello", "well hello there, hello!"),
        (r"[a-z]+", "abc 123 def ghi"),
        (r"\w+", "hi, re2 123"),
    ]

    for pattern, text in examples:
        print(f"pattern={pattern!r} text={text!r}")
        print(findall(pattern, text))


if __name__ == "__main__":
    main()
