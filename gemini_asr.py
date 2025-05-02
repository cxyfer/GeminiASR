import os
import re
import time
import argparse
import tempfile
import logging
import random
import threading
import concurrent.futures

import moviepy.editor as mp
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEYS = [key.strip() for key in os.getenv('GOOGLE_API_KEY', '').split(',') if key.strip()]
if not GOOGLE_API_KEYS:
    raise ValueError("未找到有效的 GOOGLE_API_KEY 環境變數")

# 初始化時使用第一個 KEY 進行配置
genai.configure(
    api_key=GOOGLE_API_KEYS[0]
)

# 為確保多執行緒環境中的執行緒安全，創建一個執行緒本地存儲
thread_local = threading.local()

# 建立日誌鎖，確保日誌輸出互斥
log_lock = threading.RLock()

# 新增一個執行緒安全的鎖，用於管理 API KEY 列表
api_keys_lock = threading.RLock()

# 保存已被限流的 API 金鑰
exhausted_keys = set()

def get_random_api_key():
    """從 API KEY 列表中隨機選取一個可用的金鑰"""
    with api_keys_lock:
        # 篩選出未被限流的金鑰
        available_keys = [key for key in GOOGLE_API_KEYS if key not in exhausted_keys]
        if not available_keys:
            logging.error("所有 API KEY 已被限流，無法繼續處理")
            raise ValueError("所有 API KEY 已被限流")
        return random.choice(available_keys)

def remove_exhausted_key(key):
    """將已限流的 API KEY 標記為不可用"""
    with api_keys_lock:
        if key in GOOGLE_API_KEYS and key not in exhausted_keys:
            exhausted_keys.add(key)
            remaining = len(GOOGLE_API_KEYS) - len(exhausted_keys)
            logging.warning(f"API KEY (後6位: ...{key[-6:]}) 已被限流，從可用池中移除。剩餘可用金鑰: {remaining}")
            return True
        return False

def setup_logging(level=logging.INFO):
    """設定日誌格式和級別"""
    # 創建自定義日誌處理器，確保多執行緒環境中輸出互斥
    class ThreadSafeHandler(logging.StreamHandler):
        def emit(self, record):
            with log_lock:
                super().emit(record)
    
    # 顏色定義
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 綠色
        'WARNING': '\033[33m',    # 黃色
        'ERROR': '\033[31m',      # 紅色
        'CRITICAL': '\033[41m',   # 紅色背景
        'RESET': '\033[0m'        # 重置顏色
    }
    
    # 自定義格式化器，根據日誌級別設定顏色
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
                record.msg = f"{COLORS[levelname]}{record.msg}{COLORS['RESET']}"
            return super().format(record)
    
    # 移除現有的處理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 建立新的執行緒安全處理器
    handler = ThreadSafeHandler()
    formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    
    # 設定日誌根處理器
    logging.root.setLevel(level)
    logging.root.addHandler(handler)
    
    logging.debug(f"日誌系統已初始化，級別: {logging.getLevelName(level)}")

def split_media(media_path, temp_dir, duration=300):
    """將視頻或音訊文件分割成較小的區段"""
    file_type = "影片" if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) else "音訊"
    logging.debug(f"開始分割{file_type} {media_path}，分段時長: {duration} 秒")
    
    # 根據文件類型載入媒體
    if file_type == "影片":
        media = mp.VideoFileClip(media_path)
        is_video = True
    else:
        media = mp.AudioFileClip(media_path)
        is_video = False
    
    total_duration = int(media.duration)
    parts = total_duration // duration if total_duration % duration == 0 else total_duration // duration + 1
    logging.debug(f"{file_type}總時長: {total_duration} 秒，將分為 {parts} 個部分")
    
    for idx in range(1, parts + 1):
        chunk_filename = os.path.join(temp_dir, f"chunk_{idx:02d}.mp3")
        logging.debug(f"處理第 {idx}/{parts} 部分 → {chunk_filename}")
        clip = media.subclip((idx - 1) * duration, min(idx * duration, total_duration))
        
        # 根據媒體類型選擇適當的保存方法
        if is_video:
            clip.audio.write_audiofile(chunk_filename, verbose=False, logger=None)
        else:
            clip.write_audiofile(chunk_filename, verbose=False, logger=None)
    
    logging.info(f"已將{file_type}分割成 {parts} 個部分")
    media.close()

def save_raw_transcript(transcript_text, output_path, chunk_name):
    """
    保存 LLM 生成的原始轉錄結果
    
    Args:
        transcript_text (str): 原始轉錄文字
        output_path (str): 目標資料夾路徑
        chunk_name (str): 音訊區塊名稱
    """
    # 確保輸出目錄存在
    os.makedirs(output_path, exist_ok=True)
    
    # 生成輸出檔案名稱
    output_file = os.path.join(output_path, f"{os.path.splitext(chunk_name)[0]}_raw.txt")
    
    # 寫入檔案
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)
    
    logging.debug(f"已保存原始轉錄結果到: {output_file}")
    return output_file

def get_transcription_prompt(extra_prompt=None):
    """
    建立轉錄提示詞模板，支援額外的提示詞
    
    Args:
        extra_prompt (str, optional): 額外的提示詞內容
    """
    template = """請將這段音訊轉錄成文字，並包含時間戳。

每個時間戳的格式應為：[MM:SS.ss] 或 [HH:MM:SS.ss]，請務必包含小數點後的秒數，精確到小數點後兩位。
例如：
[00:01.25] 這是第一句話。（表示 1 秒 250 毫秒）
[00:05.78] 這是第二句話。（表示 5 秒 780 毫秒）
[01:23.45] 這是第三句話。（表示 1 分 23 秒 450 毫秒）

每個句子不要太長，以便用於字幕。每句話應該有明確的時間戳，反映說話的實際開始時間。
**重要規則：**
1.  每行字幕的文字內容（不包含時間戳）最多不應超過 50 個中文字。
2.  請移除每行文字末尾的標點符號（例如句號、逗號、問號、驚嘆號）。

如果有音樂或聲音效果，請標註如：
[01:02.35] [音樂] 或 [01:02.35] [音效]

請使用以下語言進行轉錄：{language}"""
    
    # 如果有額外提示詞，則加入
    if extra_prompt:
        template += f"\n\n此外，以下是一些額外的提示詞，請參考：\n{extra_prompt}"
    
    return PromptTemplate.from_template(template)

def process_single_file(file, idx, duration, lang, model_name, save_raw, raw_dir, extra_prompt=None, time_offset=0, preview=False, max_retries=3):
    """
    處理單個音訊檔案的轉錄工作，設計為可在多執行緒環境中運行
    
    Args:
        file (str): 音訊檔案路徑
        idx (int): 檔案索引
        duration (int): 分段時長（秒）
        lang (str): 語言代碼
        model_name (str): Gemini 模型名稱
        save_raw (bool): 是否保存原始轉錄結果
        raw_dir (str): 原始轉錄儲存目錄
        extra_prompt (str, optional): 額外的提示詞內容
        time_offset (int, optional): 時間偏移量（秒）
        preview (bool, optional): 是否顯示原始轉錄結果預覽
        max_retries (int, optional): 最大重試次數，預設為3
        
    Returns:
        tuple: (SRT 格式的字幕內容, 原始轉錄檔案路徑)
    """
    basename = os.path.basename(file)
    logging.info(f"正在轉錄 {basename} (索引 {idx})...")
    logging.debug(f"應用時間偏移量: {time_offset} 秒")
    time1 = time.time()
    
    # 設定提示詞模板
    prompt_template = get_transcription_prompt(extra_prompt)
    
    # 準備提示詞
    prompt = prompt_template.format(language=lang)
    logging.debug(f"已生成提示詞模板，語言設定: {lang}")
    
    # 讀取音訊檔案
    try:
        with open(file, "rb") as f:
            audio_data = f.read()
        logging.debug(f"已讀取音訊檔案 {file}，大小: {len(audio_data)/1024/1024:.2f} MB")
    except Exception as e:
        logging.error(f"讀取音訊檔案 {file} 時發生錯誤: {e}")
        return None, None
        
    # 設定 Gemini 模型配置
    generation_config = {
        "temperature": 0,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": None
    }
    logging.debug("轉錄設定: temperature=0, max_tokens=None")
    
    # 實作重試邏輯
    retries = 0
    last_error = None
    
    while retries <= max_retries:
        current_key = None
        try:
            # 隨機選取一個 API KEY 並配置
            try:
                current_key = get_random_api_key()
                logging.debug(f"使用隨機選取的 API KEY (後6位: ...{current_key[-6:]})")
                
                # 在當前執行緒中設定 API KEY
                genai.configure(api_key=current_key)
            except ValueError as e:
                # 已經沒有可用的 API KEY
                logging.error(f"無法獲取可用的 API KEY: {e}")
                return None, None
            
            # 創建模型並送出請求
            model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
            
            response = model.generate_content([
                prompt,
                {"mime_type": "audio/mp3", "data": audio_data}
            ])
            
            # 獲取原始轉錄文字
            raw_transcript = response.text
            logging.debug(f"已收到 Gemini API 回應，回應長度: {len(raw_transcript)} 字元")
            
            # 印出原始轉錄結果 (可能較長，只顯示前200個字符和後200個字符)
            if preview:
                if len(raw_transcript) > 400:
                    preview_text = f"{raw_transcript[:200]}...\n...\n{raw_transcript[-200:]}"
                else:
                    preview_text = raw_transcript
                
                logging.info(f"原始轉錄結果預覽:\n{preview_text}")
            
            # 保存原始轉錄結果
            raw_file = None
            if save_raw:
                raw_file = save_raw_transcript(raw_transcript, raw_dir, basename)
                logging.info(f"原始轉錄結果已保存至: {raw_file}")
            
            # 直接將 Gemini 的回應轉換為 SRT 格式
            logging.debug(f"處理轉錄結果，時間偏移: {time_offset} 秒")
            srt_content = direct_to_srt(raw_transcript, time_offset)
            
            if not srt_content:
                logging.error(f"轉錄 {file} 失敗")
                return None, raw_file
                
            # 使用字幕編號計數而不是換行符
            subtitle_count = srt_content.count("\n\n") if srt_content else 0
            logging.debug(f"已將轉錄結果轉換為 SRT 格式，估計字幕數量: {subtitle_count}") 
            
            time2 = time.time()
            processing_time = time2 - time1
            logging.info(f"已完成 {basename} 的轉錄，耗時 {processing_time:.2f} 秒")
            
            return srt_content, raw_file
            
        except Exception as e:
            last_error = e
            error_message = str(e)
            
            # 檢查是否為配額限制錯誤 (429)
            if "429" in error_message and current_key:
                logging.warning(f"API KEY 限流錯誤: {error_message}")
                # 標記當前 KEY 為已用盡
                if remove_exhausted_key(current_key):
                    retries -= 1  # 如果成功移除限流金鑰，不計入重試次數
                
            retries += 1
            
            if retries <= max_retries:
                backoff_time = 2 ** retries  # 指數退避策略
                logging.warning(f"第 {retries} 次重試，等待 {backoff_time} 秒...")
                time.sleep(backoff_time)
            else:
                logging.error(f"處理 {file} 時發生錯誤，已重試 {max_retries} 次: {last_error}")
                
    # 所有重試都失敗
    logging.error(f"處理 {file} 時發生錯誤: {last_error}", exc_info=True)
    return None, None

def transcribe_with_gemini(temp_dir, duration=300, **kwargs):
    lang = kwargs.get("lang", 'zh-TW')
    model_name = kwargs.get("model", "gemini-2.5-pro-exp-03-25")
    save_raw = kwargs.get("save_raw", False)
    raw_dir = kwargs.get("raw_dir", None)
    original_file = kwargs.get("original_file", "unknown")
    max_workers = kwargs.get("max_workers", min(32, (os.cpu_count() or 1) * 5, len(GOOGLE_API_KEYS)))
    extra_prompt = kwargs.get("extra_prompt", None)
    time_offset = kwargs.get("time_offset", 0)  # 獲取時間偏移量，預設為0
    preview = kwargs.get("preview", False)  # 新預設為 False
    
    # 如果未指定原始轉錄儲存目錄，則使用預設路徑
    if raw_dir is None:
        # 從原始檔案路徑獲取目錄和檔名
        base_dir = os.path.dirname(original_file)
        base_name = os.path.splitext(os.path.basename(original_file))[0]
        raw_dir = os.path.join(base_dir, f"{base_name}_transcripts")
    
    logging.debug(f"開始轉錄處理，語言: {lang}，模型: {model_name}，最大工作執行緒數: {max_workers}，時間偏移量: {time_offset}秒")
    files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".mp3")]
    files.sort()  # 確保檔案順序
    logging.debug(f"找到 {len(files)} 個待轉錄的音訊檔案")
    
    # 創建結果儲存容器
    transcripts = [None] * len(files)  # 預先分配空間
    raw_transcripts = []
    
    # 確保輸出目錄存在
    if save_raw:
        os.makedirs(raw_dir, exist_ok=True)
    
    # 使用 ThreadPoolExecutor 並行處理音訊檔案
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 建立任務列表
        future_to_idx = {}
        for idx, file in enumerate(files):
            # 計算每個片段的時間偏移
            segment_time_offset = time_offset + (idx * duration)
            logging.debug(f"檔案 {idx} 的時間偏移量: {segment_time_offset}秒")
            
            future = executor.submit(
                process_single_file, 
                file, idx, duration, lang, model_name, 
                save_raw, raw_dir, extra_prompt, segment_time_offset, preview  # 傳入預覽參數
            )
            future_to_idx[future] = idx
        
        # 收集結果，確保按順序儲存
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                srt_content, raw_file = future.result()
                transcripts[idx] = srt_content
                if raw_file:
                    raw_transcripts.append(raw_file)
            except Exception as e:
                logging.error(f"獲取索引 {idx} 的結果時發生錯誤: {e}", exc_info=True)
    
    # 過濾掉失敗的轉錄結果
    transcripts = [t for t in transcripts if t is not None]
    
    logging.debug(f"所有音訊檔案轉錄完成，共 {len(transcripts)} 個成功轉錄結果")
    
    # 將原始轉錄結果合併為一個檔案
    if save_raw and raw_transcripts:
        combined_raw_file = os.path.join(raw_dir, "combined_raw.txt")
        # 按照檔案名稱排序，確保正確的順序
        raw_transcripts.sort()
        with open(combined_raw_file, "w", encoding="utf-8") as f:
            for idx, raw_file in enumerate(raw_transcripts):
                with open(raw_file, "r", encoding="utf-8") as rf:
                    f.write(f"=== 區塊 {idx+1} ===\n")
                    f.write(rf.read())
                    f.write("\n\n")
        logging.info(f"已合併所有原始轉錄結果至: {combined_raw_file}")
    
    return transcripts

def direct_to_srt(transcript_text, time_offset=0):
    """
    直接將 Gemini 轉錄結果轉換為 SRT 格式。
    
    Args:
        transcript_text (str): Gemini 轉錄的文字
        time_offset (int): 時間偏移量（秒）
        
    Returns:
        str: SRT 格式的字幕內容
    """
    try:
        logging.debug(f"開始將轉錄文字轉換為 SRT 格式，時間偏移: {time_offset} 秒")
        lines = transcript_text.strip().splitlines()
        logging.debug(f"轉錄文字包含 {len(lines)} 行")
        
        srt_lines = []
        srt_index = 1
        
        # 匹配時間戳和文字內容 - 支援精確時間戳（含毫秒）
        # 匹配格式: [HH:MM:SS.ss] 或 [MM:SS.ss]
        line_regex = re.compile(r'^\[((?:\d{2}:)?\d{2}:\d{2}(?:\.\d+)?)\]\s*(.+)$')
        
        matched_count = 0
        skipped_count = 0
        
        # 存儲前一個時間戳，用於計算持續時間
        prev_timestamp_seconds = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                skipped_count += 1
                continue
                
            match = line_regex.match(line)
            if not match:
                logging.debug(f"第 {i+1} 行不符合時間戳格式: {line[:50] + ('...' if len(line) > 50 else '')}")
                skipped_count += 1
                continue
                
            timestamp, content = match.groups()
            seconds = timestamp_to_seconds(timestamp)
            
            if seconds is None:
                logging.warning(f"第 {i+1} 行時間戳解析失敗: {timestamp}")
                skipped_count += 1
                continue
                
            # 應用時間偏移
            seconds += time_offset
            
            # 計算結束時間
            # 1. 檢查下一行是否存在，如果存在並有有效時間戳，則使用下一行時間戳作為結束時間
            # 2. 否則使用預設持續時間 (通常為 3-5 秒)
            next_timestamp_seconds = None
            default_duration = 3.0  # 預設持續時間（秒）
            
            # 尋找下一個有效時間戳
            for next_line in lines[i+1:]:
                next_match = line_regex.match(next_line.strip())
                if next_match:
                    next_timestamp, _ = next_match.groups()
                    next_timestamp_seconds = timestamp_to_seconds(next_timestamp)
                    if next_timestamp_seconds is not None:
                        next_timestamp_seconds += time_offset
                        break
            
            # 決定結束時間
            if next_timestamp_seconds is not None:
                end_time = next_timestamp_seconds
                # 確保持續時間至少為 0.5 秒
                if end_time - seconds < 0.5:
                    end_time = seconds + 0.5
            else:
                # 如果沒有下一個時間戳，使用預設持續時間
                end_time = seconds + default_duration
            
            # 確保時間戳包含小數部分，以保持毫秒精度
            start_formatted = format_time_srt(seconds)
            end_formatted = format_time_srt(end_time)
            
            srt_lines.append(f"{srt_index}")
            srt_lines.append(f"{start_formatted} --> {end_formatted}")
            srt_lines.append(f"{content}")
            srt_lines.append("")  # 空行分隔
            
            srt_index += 1
            matched_count += 1
            
            # 更新前一個時間戳
            prev_timestamp_seconds = seconds
            
        logging.debug(f"SRT 轉換完成: 總行數={len(lines)}, 成功匹配={matched_count}, 已跳過={skipped_count}")
        return "\n".join(srt_lines)
    except Exception as e:
        logging.error(f"轉換為 SRT 格式時發生錯誤: {e}", exc_info=True)
        return None

def timestamp_to_seconds(ts_str):
    """
    將 HH:MM:SS.ss 或 MM:SS.ss 格式的時間戳字串轉換為總秒數。
    支援毫秒精度。
    
    Args:
        ts_str (str): HH:MM:SS.ss 或 MM:SS.ss 格式的時間戳字串
        
    Returns:
        float or None: 總秒數（包含小數部分），如果解析失敗則返回 None
    """
    try:
        # 檢查是否包含小數點
        ms_part = 0.0
        if '.' in ts_str:
            main_part, ms_part_str = ts_str.split('.')
            ms_part = float('0.' + ms_part_str)
        else:
            main_part = ts_str
            
        # 將時間戳分割為各部分
        parts = list(map(int, main_part.split(':')))
        
        if len(parts) == 3:  # HH:MM:SS 格式
            h, m, s = parts
            return h * 3600 + m * 60 + s + ms_part
        elif len(parts) == 2:  # MM:SS 格式
            m, s = parts
            return m * 60 + s + ms_part
        else:
            # 部分數量無效
            return None
    except (ValueError, AttributeError, IndexError) as e:
        logging.debug(f"時間戳解析失敗: {ts_str}, 錯誤: {e}")
        return None

def format_time_srt(seconds):
    """
    將秒數格式化為 SRT 格式的時間戳（HH:MM:SS,mmm）。
    
    Args:
        seconds (float): 秒數
        
    Returns:
        str: SRT 格式的時間戳
    """
    if seconds is None or seconds < 0:
        seconds = 0.0  # 如果輸入無效或為負，預設為 0
        
    # 計算小時、分鐘和秒
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    # 取得毫秒部分（保留 3 位精度）
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def combine_subtitles(subtitles):
    """
    簡單合併多個字幕檔案，保持正確的索引編號。
    
    Args:
        subtitles (list): 字幕內容的列表
        
    Returns:
        str: 合併後的字幕內容
    """
    logging.debug(f"開始合併 {len(subtitles)} 個字幕檔案")
    result = []
    current_idx = 1
    total_entries = 0
    
    for subtitle_idx, subtitle in enumerate(subtitles):
        logging.debug(f"處理第 {subtitle_idx+1} 個字幕檔案，長度: {len(subtitle)} 字元")
        lines = subtitle.splitlines()
        i = 0
        subtitle_entries = 0
        
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
                
            # 檢查是否為索引行（純數字）
            if lines[i].strip().isdigit() and i + 2 < len(lines):
                # 替換索引為當前正確的索引
                result.append(str(current_idx))
                result.append(lines[i+1])  # 時間行
                result.append(lines[i+2])  # 文本行
                result.append("")          # 空行
                current_idx += 1
                subtitle_entries += 1
                i += 3
            else:
                i += 1
        
        total_entries += subtitle_entries
        logging.debug(f"第 {subtitle_idx+1} 個字幕檔案處理完成，包含 {subtitle_entries} 個字幕條目")
    
    logging.info(f"字幕合併完成，總計 {total_entries} 個字幕條目")
    return "\n".join(result)

def main(video_path, skip_existing=False, **kwargs):
    duration = kwargs.get("duration", 300)
    lang = kwargs.get("lang", 'zh-TW')
    model = kwargs.get("model", "gemini-2.5-pro-exp-03-25")
    save_raw = kwargs.get("save_raw", False)
    max_workers = kwargs.get("max_workers", min(32, (os.cpu_count() or 1) * 5, len(GOOGLE_API_KEYS)))
    extra_prompt = kwargs.get("extra_prompt", None)
    time_offset = kwargs.get("time_offset", 0)  # 獲取時間偏移量，預設為0

    logging.debug(f"處理參數: 分段時長={duration}秒, 語言={lang}, 模型={model}, 保存原始轉錄={save_raw}, 最大工作執行緒數={max_workers}, 跳過已存在={skip_existing}, 時間偏移量={time_offset}秒")

    _, ext = os.path.splitext(video_path)
    output_file = video_path.replace(ext, ".srt")

    # Check if SRT file exists and skip if requested
    if skip_existing and os.path.exists(output_file):
        logging.info(f"字幕檔案 '{output_file}' 已存在，跳過處理 '{video_path}'")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.debug(f"創建臨時目錄: {temp_dir}")
        if ext.lower() in [".mp4", ".avi", ".mkv"]:
            logging.info("檢測到影片檔案，開始分割音訊")
            split_media(video_path, temp_dir, duration=duration)
        elif ext.lower() in [".mp3", ".wav"]:
            logging.info("檢測到音訊檔案，開始分割")
            split_media(video_path, temp_dir, duration=duration)
        else:
            logging.error(f"不支援的檔案格式: {ext}")
            return

        logging.info("開始多執行緒轉錄處理...")
        subs = transcribe_with_gemini(temp_dir, duration=duration, lang=lang, model=model,
                                    save_raw=save_raw, original_file=video_path,
                                    max_workers=max_workers,
                                    extra_prompt=extra_prompt,
                                    time_offset=time_offset)  # 傳遞時間偏移量

        if not subs:
            logging.error("轉錄失敗，未獲得任何字幕")
            return

        logging.info("開始合併字幕...")
        combined_subs = combine_subtitles(subs)

        logging.debug(f"寫入字幕檔案: {output_file}")
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(combined_subs)
        logging.info(f"字幕已儲存至 {output_file}")

def clip(filepath, st=None, ed=None):
    """
    剪輯影片的指定部分並提取音訊。
    
    Args:
        filepath (str): 影片檔案路徑
        st (int, optional): 開始時間（秒）
        ed (int, optional): 結束時間（秒）
        
    Returns:
        str: 輸出的音訊檔案路徑
    """
    logging.info(f"準備剪輯影片: {filepath}")
    logging.debug(f"剪輯範圍: 開始={st if st is not None else '開頭'}, 結束={ed if ed is not None else '結尾'}")
    
    clip = mp.VideoFileClip(filepath)
    logging.debug(f"原始影片時長: {clip.duration:.2f} 秒")
    
    if st is not None or ed is not None:
        if st is None:
            st = 0
        if ed is None:
            ed = int(clip.duration)
        logging.info(f"剪輯影片，範圍: {st}-{ed} 秒")
        clip = clip.subclip(st, ed)
        newpath = filepath.replace(".mp4", f"_{st}-{ed}.mp3")
    else:
        logging.info("提取完整影片的音訊")
        newpath = filepath.replace(".mp4", ".mp3")
        
    logging.debug(f"開始提取音訊到: {newpath}")
    clip.audio.write_audiofile(newpath, verbose=False, logger=None)
    logging.info(f"音訊提取完成: {newpath}")
    clip.close()
    return newpath

def clip_and_transcribe(filepath, st=None, ed=None, skip_existing=False, **kwargs):
    """
    剪輯影片並轉錄。

    Args:
        filepath (str): 影片檔案路徑
        st (int, optional): 開始時間（秒）
        ed (int, optional): 結束時間（秒）
        skip_existing (bool): 如果 SRT 已存在，是否跳過
    """
    logging.info(f"開始剪輯並轉錄: {filepath}")

    # Check based on the original filepath's expected SRT name
    _, ext = os.path.splitext(filepath)
    output_srt_path = filepath.replace(ext, ".srt")
    if skip_existing and os.path.exists(output_srt_path):
        logging.info(f"字幕檔案 '{output_srt_path}' 已存在，跳過剪輯和轉錄 '{filepath}'")
        return

    if st is None:
        st = 0
    newpath = clip(filepath, st, ed)
    logging.debug(f"開始轉錄剪輯後的音訊: {newpath}")
    
    # 將原始開始時間作為參數傳遞，用於時間戳校正
    kwargs['time_offset'] = st
    logging.debug(f"設定時間偏移量: {st} 秒，用於校正字幕時間戳")
    
    # Pass skip_existing=False here, as the check was already done based on the *original* filename
    # Or pass skip_existing=skip_existing if main should re-check based on the newpath (less likely desired)
    main(newpath, **kwargs) # Pass kwargs explicitly


def process_directory(directory_path, skip_existing=False, **kwargs):
    """
    處理目錄中的所有影片和音訊檔案。

    Args:
        directory_path (str): 目錄路徑
        skip_existing (bool): 如果 SRT 已存在，是否跳過
        **kwargs: 傳遞給 main 或 clip_and_transcribe 函數的參數
    """
    logging.info(f"處理目錄: {directory_path}")

    # 支援的檔案類型
    video_extensions = [".mp4", ".avi", ".mkv"]
    audio_extensions = [".mp3", ".wav"]
    supported_extensions = video_extensions + audio_extensions

    # 獲取目錄中的所有檔案
    files = []
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            _, ext = os.path.splitext(filename)
            if ext.lower() in supported_extensions:
                # Avoid processing generated SRT files
                if ext.lower() != ".srt":
                    files.append(filepath)

    if not files:
        logging.warning("目錄中沒有找到支援的影片或音訊檔案")
        return

    logging.info(f"在目錄中找到 {len(files)} 個支援的檔案")

    # 依次處理每個檔案
    for i, filepath in enumerate(files):
        logging.info(f"處理檔案 {i+1}/{len(files)}: {filepath}")

        # Determine the expected SRT path for the current file
        _, ext = os.path.splitext(filepath)
        output_srt_path = filepath.replace(ext, ".srt")

        # Check if skipping is needed
        if skip_existing and os.path.exists(output_srt_path):
            logging.info(f"字幕檔案 '{output_srt_path}' 已存在，跳過處理 '{filepath}'")
            continue # Skip to the next file

        # Extract relevant kwargs for main/clip_and_transcribe
        common_kwargs = {
            "duration": kwargs.get("duration", 300),
            "lang": kwargs.get("lang", 'zh-TW'),
            "model": kwargs.get("model", "gemini-2.5-pro-exp-03-25"),
            "save_raw": kwargs.get("save_raw"),
            "max_workers": kwargs.get("max_workers", min(32, (os.cpu_count() or 1) * 5)),
            "extra_prompt": kwargs.get("extra_prompt"),
        }

        # 檢查是否需要剪輯
        start_time = kwargs.get("start")
        end_time = kwargs.get("end")
        if start_time is not None or end_time is not None:
            # 確保開始時間有值，用於時間戳校正
            if start_time is None:
                start_time = 0
            logging.debug(f"將使用時間偏移量 {start_time} 秒進行轉錄")
            clip_and_transcribe(filepath, st=start_time, ed=end_time,
                               skip_existing=skip_existing,
                               **common_kwargs)
        else:
            main(filepath, **common_kwargs)

    logging.info("目錄處理完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Google Gemini 為影片或音訊生成字幕")
    parser.add_argument("-i", "--input", help="輸入的影片、音訊檔案或包含媒體檔案的資料夾", required=True)
    parser.add_argument("-d", "--duration", help="每個分段的時長（秒）", type=int, default=900)
    parser.add_argument("-l", "--lang", help="語言代碼", default="zh-TW")
    parser.add_argument("-m", "--model", help="Gemini 模型", default="gemini-2.5-pro-exp-03-25")
    parser.add_argument("--start", help="開始時間（秒）", type=int)
    parser.add_argument("--end", help="結束時間（秒）", type=int)
    parser.add_argument("--save-raw", help="保存原始轉錄結果", action="store_true")
    parser.add_argument("--skip-existing", help="如果 SRT 字幕檔案已存在，則跳過處理", action="store_true")
    parser.add_argument("--debug", help="啟用 DEBUG 級別日誌", action="store_true")
    parser.add_argument("--max-workers", help="最大工作執行緒數 (預設情況下不能超過 GOOGLE_API_KEYS 的數量)", type=int, default=min(32, (os.cpu_count() or 1) * 5))
    parser.add_argument("--extra-prompt", help="額外的提示詞 或 包含提示詞的檔案路徑", type=str)
    parser.add_argument("--ignore-keys-limit", help="忽略 API KEY 數量對最大工作執行緒數的限制", action="store_true")
    parser.add_argument("--preview", help="顯示原始轉錄結果預覽", action="store_true")
    args = parser.parse_args()

    # Set logging level
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    logging.info("Gemini ASR 轉錄服務啟動")
    logging.info(f"已載入 {len(GOOGLE_API_KEYS)} 個 API KEY")
    if not args.ignore_keys_limit:
        args.max_workers = min(args.max_workers, len(GOOGLE_API_KEYS))
    logging.info(f"並行轉錄設定: 最大工作執行緒數={args.max_workers}")

    extra_prompt_value = args.extra_prompt
    if extra_prompt_value and os.path.isfile(extra_prompt_value):
        try:
            with open(extra_prompt_value, 'r', encoding='utf-8') as f:
                extra_prompt_value = f.read().strip()
            logging.info(f"已從檔案 '{args.extra_prompt}' 讀取額外提示詞。")
        except Exception as e:
            logging.warning(f"無法讀取提示詞檔案 '{args.extra_prompt}': {e}。將忽略額外提示詞。")
            extra_prompt_value = None
    if extra_prompt_value:
        logging.info(f"使用額外提示詞：\n{extra_prompt_value}")

    # Create a dictionary of arguments to pass to functions
    func_kwargs = {
        "duration": args.duration,
        "lang": args.lang,
        "model": args.model,
        "save_raw": args.save_raw,
        "max_workers": args.max_workers,
        "extra_prompt": extra_prompt_value,
        "skip_existing": args.skip_existing,
        "preview": args.preview
    }

    # Check if the input is a directory
    if os.path.isdir(args.input):
        logging.info("輸入是資料夾，將處理資料夾中的所有支援媒體檔案")
        process_directory(args.input, start=args.start, end=args.end, **func_kwargs) # Pass skip_existing via func_kwargs
    else:
        # Original logic for processing a single file
        if args.start is not None or args.end is not None:
            clip_and_transcribe(args.input, st=args.start, ed=args.end, **func_kwargs) # Pass skip_existing via func_kwargs
        else:
            main(args.input, **func_kwargs) # Pass skip_existing via func_kwargs

    logging.info("處理完成")
