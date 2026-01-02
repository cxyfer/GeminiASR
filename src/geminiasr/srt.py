import logging
import re

logger = logging.getLogger("geminiasr")

CHUNK_DURATION_TOLERANCE = 5


def direct_to_srt(transcript_text: str, time_offset: int = 0, chunk_duration: int | None = None) -> str | None:
    try:
        logger.debug("開始將轉錄文字轉換為 SRT 格式，時間偏移: %s 秒", time_offset)
        lines = transcript_text.strip().splitlines()
        logger.debug("轉錄文字包含 %s 行", len(lines))

        line_regex = re.compile(r"^\[((?:\d{2}:)?\d{2}:\d{2}(?:\.\d+)?)\]\s*(.+)$")
        skipped_count = 0
        out_of_range_count = 0
        valid_entries: list[tuple[float, str]] = []

        # Phase 1: 解析並過濾有效行
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                skipped_count += 1
                continue

            match = line_regex.match(line)
            if not match:
                logger.debug(
                    "第 %s 行不符合時間戳格式: %s",
                    i + 1,
                    line[:50] + ("..." if len(line) > 50 else ""),
                )
                skipped_count += 1
                continue

            timestamp, content = match.groups()
            seconds = timestamp_to_seconds(timestamp)
            if seconds is None:
                logger.warning("第 %s 行時間戳解析失敗: %s", i + 1, timestamp)
                skipped_count += 1
                continue

            if chunk_duration is not None and seconds > chunk_duration + CHUNK_DURATION_TOLERANCE:
                logger.warning(
                    "第 %s 行時間戳超出範圍 (%.2f 秒 > chunk 時長 %s 秒): [%s] %s",
                    i + 1,
                    seconds,
                    chunk_duration,
                    timestamp,
                    content[:30] + ("..." if len(content) > 30 else ""),
                )
                out_of_range_count += 1
                skipped_count += 1
                continue

            valid_entries.append((seconds + time_offset, content))

        # Phase 2: 基於有效行生成 SRT
        srt_lines: list[str] = []
        default_duration = 3.0
        for idx, (seconds, content) in enumerate(valid_entries):
            if idx + 1 < len(valid_entries):
                end_time = valid_entries[idx + 1][0]
                if end_time - seconds < 0.5:
                    end_time = seconds + 0.5
            else:
                end_time = seconds + default_duration
            # 限制 end_time 不超出 chunk 邊界
            if chunk_duration is not None:
                max_end = chunk_duration + time_offset + CHUNK_DURATION_TOLERANCE
                if end_time > max_end:
                    end_time = max_end

            srt_lines.append(f"{idx + 1}")
            srt_lines.append(f"{format_time_srt(seconds)} --> {format_time_srt(end_time)}")
            srt_lines.append(f"{content}")
            srt_lines.append("")

        if out_of_range_count > 0:
            logger.warning(
                "SRT 轉換完成: 總行數=%s, 成功匹配=%s, 已跳過=%s (其中超出範圍=%s)",
                len(lines),
                len(valid_entries),
                skipped_count,
                out_of_range_count,
            )
        else:
            logger.debug(
                "SRT 轉換完成: 總行數=%s, 成功匹配=%s, 已跳過=%s",
                len(lines),
                len(valid_entries),
                skipped_count,
            )
        return "\n".join(srt_lines)
    except Exception as exc:
        logger.error("轉換為 SRT 格式時發生錯誤: %s", exc, exc_info=True)
        return None


def timestamp_to_seconds(ts_str: str) -> float | None:
    try:
        ms_part = 0.0
        if "." in ts_str:
            main_part, ms_part_str = ts_str.split(".")
            ms_part = float("0." + ms_part_str)
        else:
            main_part = ts_str

        parts = list(map(int, main_part.split(":")))

        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds + ms_part
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds + ms_part
        return None
    except (ValueError, AttributeError, IndexError) as exc:
        logger.debug("時間戳解析失敗: %s, 錯誤: %s", ts_str, exc)
        return None


def format_time_srt(seconds: float) -> str:
    if seconds is None or seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def combine_subtitles(subtitles: list[str]) -> str:
    logger.debug("開始合併 %s 個字幕檔案", len(subtitles))
    result: list[str] = []
    current_idx = 1
    total_entries = 0

    for subtitle_idx, subtitle in enumerate(subtitles):
        logger.debug("處理第 %s 個字幕檔案，長度: %s 字元", subtitle_idx + 1, len(subtitle))
        lines = subtitle.splitlines()
        i = 0
        subtitle_entries = 0

        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue

            if lines[i].strip().isdigit() and i + 2 < len(lines):
                result.append(str(current_idx))
                result.append(lines[i + 1])
                result.append(lines[i + 2])
                result.append("")
                current_idx += 1
                subtitle_entries += 1
                i += 3
            else:
                i += 1

        total_entries += subtitle_entries
        logger.debug("第 %s 個字幕檔案處理完成，包含 %s 個字幕條目", subtitle_idx + 1, subtitle_entries)

    logger.info("字幕合併完成，總計 %s 個字幕條目", total_entries)
    return "\n".join(result)
