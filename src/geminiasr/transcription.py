import collections
import logging
import mimetypes
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from google import genai
from google.genai import types

from .api_keys import key_manager
from .config import Config
from .prompt import get_transcription_prompt
from .srt import combine_subtitles, direct_to_srt

logger = logging.getLogger("geminiasr")


def save_raw_transcript(transcript_text: str, output_path: str, chunk_name: str) -> str:
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{os.path.splitext(chunk_name)[0]}_raw.txt")
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(transcript_text)
    logger.debug("已保存原始轉錄結果到: %s", output_file)
    return output_file


def process_single_file(
    file_path: str,
    idx: int,
    duration: int,
    lang: str,
    model_name: str,
    save_raw: bool,
    raw_dir: str,
    base_url: str,
    extra_prompt: str | None = None,
    time_offset: int = 0,
    preview: bool = False,
    max_retries: int = 3,
) -> tuple[str | None, str | None]:
    basename = os.path.basename(file_path)
    logger.info("正在轉錄 %s (索引 %s)...", basename, idx)
    logger.debug("應用時間偏移量: %s 秒", time_offset)
    time1 = time.time()

    prompt = get_transcription_prompt(lang, extra_prompt)
    logger.debug("已生成提示詞模板，語言設定: %s", lang)

    generation_config = types.GenerateContentConfig(
        temperature=0,
        top_p=1,
        top_k=32,
        max_output_tokens=None,
    )

    retries = 0
    last_error = None

    while retries <= max_retries:
        current_key = None
        try:
            try:
                current_key = key_manager.get_key()
                logger.debug("使用隨機選取的 API KEY (後6位: ...%s)", current_key[-6:])
                client = genai.Client(
                    api_key=current_key,
                    http_options=types.HttpOptions(base_url=base_url),
                )
            except ValueError as exc:
                logger.error("無法獲取可用的 API KEY: %s", exc)
                return None, None

            try:
                logger.debug("正在上傳音訊檔案 %s", file_path)
                with open(file_path, "rb") as handle:
                    file_bytes = handle.read()
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type is None:
                    raise ValueError("Failed to guess the mime type of the file.")
                uploaded_file = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
                logger.debug("已成功上傳檔案 %s", file_path)
            except Exception as exc:
                logger.error("上傳音訊檔案失敗: %s", exc)
                return None, None

            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, uploaded_file],
                config=generation_config,
            )

            raw_transcript = response.text
            logger.debug("已收到 Gemini API 回應，回應長度: %s 字元", len(raw_transcript))

            if preview:
                if len(raw_transcript) > 400:
                    preview_text = f"{raw_transcript[:200]}...\n...\n{raw_transcript[-200:]}"
                else:
                    preview_text = raw_transcript
                logger.info("原始轉錄結果預覽:\n%s", preview_text)

            raw_file = None
            if save_raw:
                raw_file = save_raw_transcript(raw_transcript, raw_dir, basename)
                logger.info("原始轉錄結果已保存至: %s", raw_file)

            logger.debug("處理轉錄結果，時間偏移: %s 秒", time_offset)
            srt_content = direct_to_srt(raw_transcript, time_offset)
            if not srt_content:
                logger.error("轉錄 %s 失敗", file_path)
                return None, raw_file

            subtitle_count = srt_content.count("\n\n") if srt_content else 0
            logger.debug("已將轉錄結果轉換為 SRT，估計字幕數量: %s", subtitle_count)

            time2 = time.time()
            logger.info("已完成 %s 的轉錄，耗時 %.2f 秒", basename, time2 - time1)
            return srt_content, raw_file

        except Exception as exc:
            logger.error("處理 %s 時發生錯誤: %s", file_path, exc)
            last_error = exc
            error_message = str(exc)

            if current_key and ("429" in error_message or "403" in error_message):
                if "429" in error_message:
                    logger.warning("API KEY 限流錯誤 (429): %s", error_message)
                    if key_manager.disable_key(current_key, reason="rate_limit"):
                        retries -= 1
                elif "403" in error_message:
                    logger.error("API KEY 被禁用 (403): %s", error_message)
                    if key_manager.disable_key(current_key, reason="banned"):
                        retries -= 1

            retries += 1
            if retries <= max_retries:
                backoff_time = 2 ** retries
                logger.warning("第 %s 次重試，等待 %s 秒...", retries, backoff_time)
                time.sleep(backoff_time)
            else:
                logger.error("處理 %s 時發生錯誤，已重試 %s 次: %s", file_path, max_retries, last_error)

    logger.error("處理 %s 時發生錯誤: %s", file_path, last_error, exc_info=True)
    return None, None


def transcribe_with_gemini(
    temp_dir: str,
    config: Config,
    original_file: str,
    time_offset: int = 0,
) -> list[str] | None:
    lang = config.transcription.lang
    model_name = config.transcription.model
    save_raw = config.transcription.save_raw
    extra_prompt = config.extra_prompt
    preview = config.transcription.preview
    max_segment_retries = config.transcription.max_segment_retries

    max_workers = config.processing.max_workers
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) * 5)
    if not config.processing.ignore_keys_limit:
        max_workers = min(max_workers, key_manager.get_available_key_count())

    if max_workers < 1:
        raise ValueError("No API keys available for transcription.")

    if save_raw:
        base_dir = os.path.dirname(original_file)
        base_name = os.path.splitext(os.path.basename(original_file))[0]
        raw_dir = os.path.join(base_dir, f"{base_name}_transcripts")
    else:
        raw_dir = ""

    logger.debug(
        "開始轉錄處理，語言: %s，模型: %s，最大工作執行緒數: %s，時間偏移量: %s 秒，片段最大重試次數: %s",
        lang,
        model_name,
        max_workers,
        time_offset,
        max_segment_retries,
    )

    all_files = [
        os.path.join(temp_dir, name)
        for name in os.listdir(temp_dir)
        if name.endswith(".mp3")
    ]
    all_files.sort()
    logger.debug("找到 %s 個待轉錄的音訊檔案", len(all_files))

    if not all_files:
        logger.info("在暫存目錄中未找到任何 .mp3 檔案可供轉錄。")
        return []

    transcripts_results: list[str | None] = [None] * len(all_files)
    raw_transcripts_paths: list[str] = []

    if save_raw:
        os.makedirs(raw_dir, exist_ok=True)

    Task = collections.namedtuple(
        "Task", ["file_path", "original_index", "current_retry_count", "segment_time_offset"]
    )

    tasks_to_process: collections.deque = collections.deque()
    duration = config.transcription.duration
    for idx, file_path in enumerate(all_files):
        segment_time_offset = time_offset + (idx * duration)
        tasks_to_process.append(Task(file_path, idx, 0, segment_time_offset))

    active_futures = {}
    overall_transcription_failed = False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while tasks_to_process or active_futures:
            while tasks_to_process and len(active_futures) < max_workers:
                task = tasks_to_process.popleft()
                logger.debug(
                    "提交任務: %s (索引 %s), 第 %s 次嘗試",
                    os.path.basename(task.file_path),
                    task.original_index,
                    task.current_retry_count + 1,
                )
                future = executor.submit(
                    process_single_file,
                    task.file_path,
                    task.original_index,
                    duration,
                    lang,
                    model_name,
                    save_raw,
                    raw_dir,
                    config.api.base_url,
                    extra_prompt,
                    task.segment_time_offset,
                    preview,
                )
                active_futures[future] = task

            if not active_futures:
                break

            done, _ = wait(active_futures.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                task = active_futures.pop(future)
                original_idx = task.original_index
                file_basename = os.path.basename(task.file_path)

                try:
                    srt_content, raw_file_path = future.result()
                    if srt_content is not None:
                        transcripts_results[original_idx] = srt_content
                        if raw_file_path:
                            raw_transcripts_paths.append(raw_file_path)
                        if task.current_retry_count > 0:
                            logger.info(
                                "片段 %s (索引 %s) 在第 %s 次嘗試後轉錄成功。",
                                file_basename,
                                original_idx,
                                task.current_retry_count + 1,
                            )
                        else:
                            logger.debug(
                                "片段 %s (索引 %s) 首次嘗試轉錄成功。",
                                file_basename,
                                original_idx,
                            )
                    else:
                        raise Exception("process_single_file 返回 None，表示轉錄失敗")

                except Exception as exc:
                    logger.warning(
                        "片段 %s (索引 %s) 第 %s 次轉錄嘗試失敗。錯誤：%s",
                        file_basename,
                        original_idx,
                        task.current_retry_count + 1,
                        exc,
                    )
                    if task.current_retry_count < max_segment_retries:
                        new_task = Task(
                            task.file_path,
                            original_idx,
                            task.current_retry_count + 1,
                            task.segment_time_offset,
                        )
                        tasks_to_process.append(new_task)
                        logger.info(
                            "準備對片段 %s (索引 %s) 進行第 %s 次重試 (共 %s 次嘗試)。",
                            file_basename,
                            original_idx,
                            new_task.current_retry_count + 1,
                            max_segment_retries + 1,
                        )
                    else:
                        logger.error(
                            "片段 %s (索引 %s) 在 %s 次嘗試後最終轉錄失敗。最後錯誤：%s。將中止整個轉錄過程。",
                            file_basename,
                            original_idx,
                            max_segment_retries + 1,
                            exc,
                        )
                        overall_transcription_failed = True
                        break

            if overall_transcription_failed:
                break

        if overall_transcription_failed:
            logger.error("由於有片段最終轉錄失敗，整體轉錄任務已中止。正在取消剩餘任務...")
            for future_to_cancel in list(active_futures.keys()):
                future_to_cancel.cancel()
                active_futures.pop(future_to_cancel, None)
            tasks_to_process.clear()
            logger.info("所有剩餘的轉錄任務已嘗試取消。")
            return None

    if not all_files:
        return []

    if any(item is None for item in transcripts_results):
        logger.error("轉錄完成，但結果列表中包含 None 值，這不符合預期。請檢查邏輯。")
        successful_transcripts = [item for item in transcripts_results if item is not None]
        if not successful_transcripts:
            logger.error("所有轉錄結果均為 None，返回 None。")
            return None
        logger.warning("返回 %s 個部分成功的轉錄結果。", len(successful_transcripts))
        return successful_transcripts

    logger.info("所有音訊檔案轉錄完成，共 %s 個成功轉錄結果。", len(transcripts_results))

    if save_raw and raw_transcripts_paths:
        combined_raw_file = os.path.join(raw_dir, "combined_raw.txt")
        raw_transcripts_paths.sort()
        with open(combined_raw_file, "w", encoding="utf-8") as handle:
            for idx, raw_file in enumerate(raw_transcripts_paths):
                try:
                    with open(raw_file, encoding="utf-8") as raw_handle:
                        handle.write(
                            f"=== 區塊來自檔案: {os.path.basename(raw_file)} (原始順序 {idx + 1}) ===\n"
                        )
                        handle.write(raw_handle.read())
                        handle.write("\n\n")
                except Exception as exc:
                    logger.error("合併原始轉錄檔案 %s 時發生錯誤: %s", raw_file, exc)
        logger.info("已合併所有原始轉錄結果至: %s", combined_raw_file)

    return [item for item in transcripts_results if item is not None]


def combine_and_save_subtitles(subtitles: list[str], output_file: str) -> None:
    logger.info("開始合併字幕...")
    combined_subs = combine_subtitles(subtitles)

    logger.debug("寫入字幕檔案: %s", output_file)
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(combined_subs)
    logger.info("字幕已儲存至 %s", output_file)
