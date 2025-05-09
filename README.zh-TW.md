<div align="center">

# 🎙️ Gemini ASR 語音轉文字工具

[English](README.md) | [简体中文](README.zh-CN.md) | **繁體中文**

*一個使用 Google Gemini API 將影片或音訊檔案轉錄為 SRT 字幕檔案的 Python 工具。*

</div>

---

# 🎙️ Gemini ASR 語音轉文字工具

A Python tool that uses Google Gemini API to transcribe video or audio files into SRT subtitle files. 支援多執行緒、檔案分割、時間戳記剪輯和自訂提示詞。

這個工具利用了 Google 先進的多模態 AI 模型 Gemini 2.5 Pro 強大的音訊處理能力，它在理解和轉錄多種語言的口語內容方面表現出色，準確率很高。

## ✨ 功能

* 🎥 支援各種影片 (mp4, avi, mkv) 和音訊 (mp3, wav) 格式。
* ✂️ 自動將長檔案分割成較小的區塊進行處理。
* 🧵 使用多執行緒進行平行處理以加速轉錄。
* 🔄 可選地輪換使用多個 Google API Key 以提高請求成功率。
* ⏱️ 生成具有精確時間戳記 (毫秒精度) 的 SRT 字幕檔案。
* 🎬 可選地剪輯影片或音訊的特定時間段進行轉錄。
* 📄 可選地儲存 Gemini API 返回的原始轉錄文字。
* 💬 支援傳遞額外的提示詞 (文字或檔案路徑) 以引導轉錄模型。
* 📁 可以處理單一檔案或整個目錄中所有支援的檔案。
* ⏩ 提供 `--skip-existing` 選項，以避免重新處理已經有字幕的檔案。
* 🐞 支援 DEBUG 模式，用於詳細的日誌輸出。
* 🌈 使用彩色日誌，方便區分不同級別的訊息。

## 🔧 安裝

### 🛠️ 環境設定

1. **安裝 Python:** 建議使用 Python 3.8 或更高版本。
2. **安裝 uv:** 如果您尚未安裝 `uv`，請參閱 [uv 官方文件](https://github.com/astral-sh/uv) 進行安裝。`uv` 是一個極快的 Python 套件安裝和管理器。

### 📦 依賴

只需使用 `uv run` 即可自動安裝和運行腳本：

```bash
uv run gemini_asr.py -i video.mp4
```

這將自動安裝所有所需的依賴並執行腳本。

### 🔑 API Key

1. **獲取 Google API Key:** 前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 獲取您的 API Key。您可以獲取多個 Key 以提高處理效率。
2. **設定環境變數:**
   * 在專案根目錄中建立一個名為 `.env` 的檔案。
   * 在 `.env` 檔案中加入您的 API Key(s)，多個 Key 使用逗號分隔：
     ```env
     GOOGLE_API_KEY=YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3
     ```

## 📋 使用方法

### ⌨️ 命令列參數

```
python gemini_asr.py [-h] -i INPUT [-d DURATION] [-l LANG] [-m MODEL]
                      [--start START] [--end END] [--save-raw]
                      [--skip-existing] [--debug]
                      [--max-workers MAX_WORKERS]
                      [--extra-prompt EXTRA_PROMPT]
                      [--ignore-keys-limit]

arguments:
  -h, --help            顯示幫助訊息並退出
  -i INPUT, --input INPUT
                        輸入影片、音訊檔案或包含媒體檔案的資料夾
  -d DURATION, --duration DURATION
                        每個片段的持續時間（秒）（預設值：900）
  -l LANG, --lang LANG  語言代碼（預設值：zh-TW）
  -m MODEL, --model MODEL
                        Gemini 模型（預設值：gemini-2.5-pro-exp-03-25）
  --start START         開始時間（秒）
  --end END             結束時間（秒）
  --save-raw            儲存原始轉錄結果
  --skip-existing       如果 SRT 字幕檔案已存在則跳過處理
  --debug               啟用 DEBUG 級別日誌記錄
  --max-workers MAX_WORKERS
                        最大工作執行緒數（預設值：不能超過 GOOGLE_API_KEY 的數量）
  --extra-prompt EXTRA_PROMPT
                        額外的提示詞或包含提示詞的檔案路徑
  --ignore-keys-limit   忽略對最大工作執行緒數的 API Key 數量限制
```

### 💡 範例

1. **基本使用 - 轉錄影片檔案:**
   ```bash
   python gemini_asr.py -i video.mp4
   ```

2. **轉錄影片並使用自訂片段持續時間 (5 分鐘):**
   ```bash
   python gemini_asr.py -i video.mp4 -d 300
   ```

3. **僅轉錄影片的某一部分 (從 60 秒到 180 秒):**
   ```bash
   python gemini_asr.py -i video.mp4 --start 60 --end 180
   ```

4. **處理目錄中的所有媒體檔案:**
   ```bash
   python gemini_asr.py -i /path/to/media/folder
   ```

5. **使用特定語言:**
   ```bash
   python gemini_asr.py -i video.mp4 -l en
   ```

6. **使用自訂提示詞改善轉錄:**
   ```bash
   python gemini_asr.py -i lecture.mp4 --extra-prompt "This is a technical lecture about machine learning."
   ```

7. **跳過已經有 SRT 字幕的檔案:**
   ```bash
   python gemini_asr.py -i /path/to/media/folder --skip-existing
   ```

8. **儲存原始轉錄結果以供後續審查:**
   ```bash
   python gemini_asr.py -i interview.mp4 --save-raw
   ```

9. **啟用調試日誌記錄以進行故障排除:**
   ```bash
   python gemini_asr.py -i video.mp4 --debug
   ```

## 🔍 音訊處理技術細節

* 🧮 **Token 使用量**: Gemini 每秒音訊使用 32 個 token (1,920 tokens/分鐘)。有關音訊處理能力的更多詳細資訊，請參閱 [Gemini 音訊文件](https://ai.google.dev/gemini-api/docs/audio)。
* 📈 **輸出 Token**: Gemini 2.5 Pro 每個請求的輸出 token 限制為 65,536 個，這會影響可處理音訊的最大持續時間。有關詳細資訊，請參閱 [Gemini 模型文件](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-preview-03-25)。
* 📊 **速率限制**: 預設模型 (`gemini-2.5-pro-exp-03-25`) 在預覽期間是免費的，但受特定限制：250,000 TPM (每分鐘 token)，1,000,000 TPD (每天 token)，5 RPM (每分鐘請求) 和 25 RPM (每分鐘請求)。有關詳細資訊，請參閱 [速率限制文件](https://ai.google.dev/gemini-api/docs/rate-limits)。
* 💰 **定價**: 付費層每百萬 token 費用為 $1.25 (≤200k token) 或 $2.50 (>200k token)。對於超過 2 小時的音訊，建議分割檔案以避免過多的 token 使用量和潛在的成本超支。有關完整的定價資訊，請參閱 [Gemini 開發者 API 定價](https://ai.google.dev/gemini-api/docs/pricing)。

## 📝 注意事項

* 腳本使用 Gemini API，該 API 有使用限制，並可能在免費層之外產生費用。
* 為了獲得最佳效能，請考慮：
  * 🔑 使用多個 API Key (在 .env 檔案中使用逗號分隔)
  * ⏱️ 根據內容複雜性調整片段持續時間 — **將片段保持在 60 分鐘以下**，以避免輸出 token 限制，並維持免費層使用者的 TPM 限制
  * 🧵 根據系統能力和可用 API Key 的數量適當配置 max-workers
  * 🚫 `--ignore-keys-limit` 選項應謹慎使用，主要供付費層使用者使用，以避免觸及免費層嚴格的 TPM 限制
* ⚠️ 如果您遇到 429 (請求過多) 錯誤，請嘗試減少 max-workers 的數量，添加更多 API Key，或升級到付費層
* 💲 付費層使用者應使用 `gemini-2.5-pro-preview-03-25` 模型，並且由於更高的 TPM 額度，可以安全地使用 `--ignore-keys-limit` 選項。

---

**其他語言:** [English](README.md) | [简体中文](README.zh-CN.md) 