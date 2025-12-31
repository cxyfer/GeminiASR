import argparse
import logging
import os

from .api_keys import key_manager
from .config import Config, ConfigManager
from .logging import setup_logging
from .media import clip_and_transcribe, process_directory, transcribe_media_file

logger = logging.getLogger("geminiasr")


def _load_extra_prompt(value: str | None) -> str | None:
    if not value:
        return None
    if os.path.isfile(value):
        try:
            with open(value, encoding="utf-8") as handle:
                return handle.read().strip()
        except Exception as exc:
            logger.warning("無法讀取提示詞檔案 '%s': %s。將忽略額外提示詞。", value, exc)
            return None
    return value


def _resolve_max_workers(config: Config) -> None:
    max_workers = config.processing.max_workers
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 5)
    if not config.processing.ignore_keys_limit:
        key_count = key_manager.get_available_key_count()
        if key_count < 1:
            raise ValueError("No API keys configured")
        max_workers = min(max_workers, key_count)
    config.processing.max_workers = max_workers
    logger.info("並行轉錄設定: 最大工作執行緒數=%s", max_workers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 Google Gemini 為影片或音訊生成字幕")
    parser.add_argument("-i", "--input", required=True, help="輸入的影片、音訊檔案或包含媒體檔案的資料夾")
    parser.add_argument("-d", "--duration", type=int, help="每個分段的時長（秒）")
    parser.add_argument("-l", "--lang", help="語言代碼")
    parser.add_argument("-m", "--model", help="Gemini 模型")
    parser.add_argument("--start", type=int, help="開始時間（秒）")
    parser.add_argument("--end", type=int, help="結束時間（秒）")
    parser.add_argument("--save-raw", action="store_true", help="保存原始轉錄結果")
    parser.add_argument("--skip-existing", action="store_true", help="如果 SRT 字幕檔案已存在，則跳過處理")
    parser.add_argument("--no-skip-existing", action="store_true", help="覆寫現有 SRT 檔案")
    parser.add_argument("--debug", action="store_true", help="啟用 DEBUG 級別日誌")
    parser.add_argument("--max-workers", type=int, help="最大工作執行緒數")
    parser.add_argument("--extra-prompt", help="額外的提示詞或包含提示詞的檔案路徑")
    parser.add_argument("--ignore-keys-limit", action="store_true", help="忽略 API KEY 數量對最大工作執行緒數的限制")
    parser.add_argument("--preview", action="store_true", help="顯示原始轉錄結果預覽")
    parser.add_argument("--max-segment-retries", type=int, help="每個音訊區塊轉錄失敗時的最大重試次數")
    parser.add_argument("--config", help="配置檔案路徑")
    return parser


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> None:
    if args.duration is not None:
        config.transcription.duration = args.duration
    if args.lang:
        config.transcription.lang = args.lang
    if args.model:
        config.transcription.model = args.model
    if args.save_raw:
        config.transcription.save_raw = True
    if args.preview:
        config.transcription.preview = True
    if args.max_segment_retries is not None:
        config.transcription.max_segment_retries = args.max_segment_retries

    if args.max_workers is not None:
        config.processing.max_workers = args.max_workers
    if args.ignore_keys_limit:
        config.processing.ignore_keys_limit = True

    if args.debug:
        config.logging.debug = True

    if args.extra_prompt:
        config.extra_prompt = args.extra_prompt

    if args.skip_existing and args.no_skip_existing:
        raise ValueError("--skip-existing and --no-skip-existing cannot be used together")
    if args.skip_existing:
        config.transcription.skip_existing = True
    if args.no_skip_existing:
        config.transcription.skip_existing = False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = ConfigManager.load_config(args.config)
        apply_cli_overrides(config, args)

        setup_logging(config.logging.debug)
        logger.info("Gemini ASR 轉錄服務啟動")

        config.extra_prompt = _load_extra_prompt(config.extra_prompt)
        if config.extra_prompt:
            logger.info("使用額外提示詞：\n%s", config.extra_prompt)

        key_manager.configure(config.api.google_api_keys)
        _resolve_max_workers(config)
        ConfigManager.log_config_details(config, logger)

        input_path = args.input
        if os.path.isdir(input_path):
            logger.info("輸入是資料夾，將處理資料夾中的所有支援媒體檔案")
            process_directory(
                input_path,
                config,
                start=args.start,
                end=args.end,
                skip_existing=config.transcription.skip_existing,
            )
        else:
            if args.start is not None or args.end is not None:
                clip_and_transcribe(
                    input_path,
                    config,
                    start=args.start,
                    end=args.end,
                    skip_existing=config.transcription.skip_existing,
                )
            else:
                transcribe_media_file(
                    input_path,
                    config,
                    skip_existing=config.transcription.skip_existing,
                )

        logger.info("處理完成")
        return 0
    except Exception as exc:
        logging.getLogger("geminiasr").error("Error: %s", exc)
        return 1
