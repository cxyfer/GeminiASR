import argparse

from geminiasr.config import ConfigManager
from geminiasr.logging import setup_logging
from geminiasr.media import transcribe_media_file


def main() -> int:
    parser = argparse.ArgumentParser(description="GeminiASR basic usage")
    parser.add_argument("input", help="Path to a media file")
    args = parser.parse_args()

    config = ConfigManager.load_config()
    setup_logging(config.logging.debug)
    transcribe_media_file(args.input, config, skip_existing=config.transcription.skip_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
