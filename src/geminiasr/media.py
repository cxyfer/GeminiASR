import logging
import os
import tempfile
from pathlib import Path

import moviepy.editor as mp

from .config import Config
from .constants import AUDIO_FILE_EXTENSIONS, VIDEO_FILE_EXTENSIONS
from .transcription import combine_and_save_subtitles, transcribe_with_gemini

logger = logging.getLogger("geminiasr")


def split_media(media_path: str, temp_dir: str, duration: int = 300) -> None:
    is_video = media_path.lower().endswith(tuple(VIDEO_FILE_EXTENSIONS))
    file_type = "影片" if is_video else "音訊"
    logger.debug("開始分割%s %s，分段時長: %s 秒", file_type, media_path, duration)

    media = mp.VideoFileClip(media_path) if is_video else mp.AudioFileClip(media_path)
    total_duration = int(media.duration)
    parts = total_duration // duration if total_duration % duration == 0 else total_duration // duration + 1
    logger.debug("%s總時長: %s 秒，將分為 %s 個部分", file_type, total_duration, parts)

    for idx in range(1, parts + 1):
        chunk_filename = os.path.join(temp_dir, f"chunk_{idx:02d}.mp3")
        logger.debug("處理第 %s/%s 部分 → %s", idx, parts, chunk_filename)
        clip = media.subclip((idx - 1) * duration, min(idx * duration, total_duration))
        if is_video:
            clip.audio.write_audiofile(chunk_filename, verbose=False, logger=None)
        else:
            clip.write_audiofile(chunk_filename, verbose=False, logger=None)

    logger.info("已將%s分割成 %s 個部分", file_type, parts)
    media.close()


def clip_media(filepath: str, start: int | None = None, end: int | None = None) -> str:
    logger.info("準備剪輯影片: %s", filepath)
    logger.debug("剪輯範圍: 開始=%s, 結束=%s", start if start is not None else "開頭", end or "結尾")

    clip = mp.VideoFileClip(filepath)
    ext = os.path.splitext(filepath)[1]
    logger.debug("原始影片時長: %.2f 秒", clip.duration)

    if start is not None or end is not None:
        if start is None:
            start = 0
        if end is None:
            end = int(clip.duration)
        logger.info("剪輯影片，範圍: %s-%s 秒", start, end)
        clip = clip.subclip(start, end)
        newpath = filepath.replace(ext, f"_{start}-{end}.mp3")
    else:
        logger.info("提取完整影片的音訊")
        newpath = filepath.replace(ext, ".mp3")

    logger.debug("開始提取音訊到: %s", newpath)
    clip.audio.write_audiofile(newpath, verbose=False, logger=None)
    logger.info("音訊提取完成: %s", newpath)
    clip.close()
    return newpath


def transcribe_media_file(
    media_path: str,
    config: Config,
    skip_existing: bool = False,
    time_offset: int = 0,
) -> None:
    path = Path(media_path)
    output_file = path.with_suffix(".srt")

    if skip_existing and output_file.exists():
        logger.info("字幕檔案 '%s' 已存在，跳過處理 '%s'", output_file, media_path)
        return

    ext = path.suffix.lower()
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.debug("創建臨時目錄: %s", temp_dir)
        if ext in VIDEO_FILE_EXTENSIONS:
            logger.info("檢測到影片檔案，開始分割音訊")
            split_media(str(path), temp_dir, duration=config.transcription.duration)
        elif ext in AUDIO_FILE_EXTENSIONS:
            logger.info("檢測到音訊檔案，開始分割")
            split_media(str(path), temp_dir, duration=config.transcription.duration)
        else:
            logger.error("不支援的檔案格式: %s", ext)
            return

        logger.info("開始多執行緒轉錄處理...")
        subs = transcribe_with_gemini(
            temp_dir,
            config,
            original_file=str(path),
            time_offset=time_offset,
        )

        if not subs:
            logger.error("轉錄失敗，未獲得任何字幕")
            return

        combine_and_save_subtitles(subs, str(output_file))


def clip_and_transcribe(
    filepath: str,
    config: Config,
    start: int | None = None,
    end: int | None = None,
    skip_existing: bool = False,
) -> None:
    logger.info("開始剪輯並轉錄: %s", filepath)

    output_srt_path = Path(filepath).with_suffix(".srt")
    if skip_existing and output_srt_path.exists():
        logger.info("字幕檔案 '%s' 已存在，跳過剪輯和轉錄 '%s'", output_srt_path, filepath)
        return

    if start is None:
        start = 0
    newpath = clip_media(filepath, start, end)
    logger.debug("開始轉錄剪輯後的音訊: %s", newpath)

    transcribe_media_file(newpath, config, skip_existing=False, time_offset=start)


def process_directory(
    directory_path: str,
    config: Config,
    start: int | None = None,
    end: int | None = None,
    skip_existing: bool = False,
) -> None:
    logger.info("處理目錄 (包含子目錄): %s", directory_path)

    supported_extensions = VIDEO_FILE_EXTENSIONS + AUDIO_FILE_EXTENSIONS
    files: list[str] = []

    for root, _, filenames in os.walk(directory_path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            _, ext = os.path.splitext(filename)
            if ext.lower() in supported_extensions and ext.lower() != ".srt":
                files.append(filepath)

    if not files:
        logger.warning("目錄及其子目錄中沒有找到支援的影片或音訊檔案")
        return

    logger.info("在目錄及其子目錄中找到 %s 個支援的檔案", len(files))

    for idx, filepath in enumerate(files):
        logger.info("處理檔案 %s/%s: %s", idx + 1, len(files), filepath)

        output_srt_path = Path(filepath).with_suffix(".srt")
        if skip_existing and output_srt_path.exists():
            logger.info("字幕檔案 '%s' 已存在，跳過處理 '%s'", output_srt_path, filepath)
            continue

        if start is not None or end is not None:
            clip_and_transcribe(
                filepath,
                config,
                start=start,
                end=end,
                skip_existing=skip_existing,
            )
        else:
            transcribe_media_file(filepath, config, skip_existing=skip_existing)

    logger.info("目錄處理完成")
