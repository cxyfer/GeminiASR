import argparse
from pathlib import Path

from google import genai


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemini file upload demo")
    parser.add_argument("file", type=Path, help="Path to a media file")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model name")
    args = parser.parse_args()

    client = genai.Client()
    uploaded = client.files.upload(file=args.file)
    result = client.models.generate_content(
        model=args.model,
        contents=[uploaded, "\n\n", "Please describe the content of this file."],
    )
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
