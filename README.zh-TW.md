<div align="center">

# 🎙️ Gemini ASR 語音轉文字工具

[English](README.md) | [简体中文](README.zh-CN.md) | **繁體中文**

*一個使用 Google Gemini API 將影片或音訊檔案轉錄為 SRT 字幕檔案的 Python 工具。*

</div>

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
* 🔗 支援代理伺服器或自訂伺服器端，如 gemini-balance。
  * 如果您想使用 gemini-balance，需要設定 `BASE_URL` 環境變數為 `https://your-custom-url.com/`。
  * 注意：如果您使用 gemini-balance，**必須關閉程式碼執行功能**。
* ⚙️ **TOML 配置支援**：綜合的配置管理系統，支援多種配置來源。
  * 📝 支援配置檔案，並在多個位置自動搜尋
  * 🔄 多來源配置合併 (命令列 > 環境變數 > TOML > 預設值)
  * 🎛️ 在單一配置檔案中輕鬆管理所有設定

## 🔧 安裝

### 🛠️ 環境設定

1. **安裝 Python:** 建議使用 Python 3.10 或更高版本。
2. **安裝 uv:** 如果您尚未安裝 `uv`，請參閱 [uv 官方文件](https://github.com/astral-sh/uv) 進行安裝。`uv` 是一個極快的 Python 套件安裝和管理器。

### 📦 安裝

**選項 A：可編輯安裝（建議開發使用）**
```bash
pip install -e .
```

接著執行：
```bash
geminiasr -i video.mp4
```

**選項 B：使用 uv 一鍵執行**

```bash
uv run gemini_asr.py -i video.mp4
```

這將自動安裝所有所需的依賴並執行腳本。

### 🔑 API Key 配置

1. **獲取 Google API Key:** 前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 獲取您的 API Key。您可以獲取多個 Key 以提高處理效率。

2. **配置方式** (選擇其中一種):

   **方式 A：TOML 配置檔案 (推薦)**
   ```bash
   # 複製範例配置檔案
   cp config.example.toml config.toml
   
   # 編輯 config.toml 並添加您的 API Key
   nano config.toml
   ```
   
   在 `config.toml` 中：
   ```toml
   [api]
   source = "gemini"  # "gemini" 或 "openai"
   google_api_keys = ["YOUR_API_KEY_1", "YOUR_API_KEY_2", "YOUR_API_KEY_3"]
   ```

   **方式 B：環境變數**
   ```bash
   # 設定環境變數，多個 Key 使用逗號分隔
   export GOOGLE_API_KEY=YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3
   ```

   **方式 C：.env 檔案**
   ```bash
   # 在專案根目錄建立 .env 檔案
   echo "GOOGLE_API_KEY=YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3" > .env
   ```

### ⚙️ 配置系統

GeminiASR 支援靈活的配置系統，優先順序如下：
1. **命令列參數** (最高優先級)
2. **環境變數**
3. **TOML 配置檔案**
4. **預設值** (最低優先級)

**配置檔案搜尋位置** (依序搜尋):
- `./config.toml` (目前目錄)
- `./.geminiasr/config.toml`
- `~/.geminiasr/config.toml`
- `~/.config/geminiasr/config.toml`

**環境變數白名單**：
- `GOOGLE_API_KEY` (以逗號分隔)
- `GEMINIASR_LANG`, `GEMINIASR_MODEL`, `GEMINIASR_DURATION`
- `GEMINIASR_MAX_WORKERS`, `GEMINIASR_IGNORE_KEYS_LIMIT`, `GEMINIASR_DEBUG`
- `GEMINIASR_SAVE_RAW`, `GEMINIASR_SKIP_EXISTING`, `GEMINIASR_PREVIEW`
- `GEMINIASR_MAX_SEGMENT_RETRIES`, `GEMINIASR_EXTRA_PROMPT`
- `GEMINIASR_API_SOURCE`
- `GEMINIASR_BASE_URL` 或 `BASE_URL`

**OpenAI 相容端點**：
- 設定 `api.source = "openai"`（或 `GEMINIASR_API_SOURCE=openai`）。
- 若 `advanced.base_url` 保持 Gemini 預設值，會自動切換為 `https://generativelanguage.googleapis.com/v1beta/openai/`。

**配置檔案範例** (`config.toml`):
```toml
# 轉錄設定
[transcription]
duration = 900           # 片段持續時間 (秒)
lang = "zh-TW"          # 語言代碼
model = "gemini-2.5-flash"  # Gemini 模型
skip_existing = true     # 跳過已有 SRT 檔案
max_segment_retries = 3  # 每個片段最大重試次數

# 處理設定
[processing]
max_workers = 24         # 最大並行執行緒數
ignore_keys_limit = true # 忽略 API Key 限制

# 日誌設定
[logging]
debug = true            # 啟用除錯日誌

# API 設定
[api]
source = "gemini"  # "gemini" 或 "openai"
google_api_keys = ["key1", "key2", "key3"]

# 進階設定
[advanced]
extra_prompt = "prompt.md"  # 提示詞檔案路徑
base_url = "https://generativelanguage.googleapis.com/"
# base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
```

## 📋 使用方法

### ⌨️ 命令列參數

```
geminiasr [-h] -i INPUT [-d DURATION] [-l LANG] [-m MODEL]
          [--start START] [--end END] [--save-raw]
          [--skip-existing | --no-skip-existing]
          [--debug] [--max-workers MAX_WORKERS]
          [--extra-prompt EXTRA_PROMPT]
          [--ignore-keys-limit] [--preview]
          [--max-segment-retries MAX_SEGMENT_RETRIES]
          [--config CONFIG]

arguments:
  -h, --help            顯示幫助訊息並退出
  -i INPUT, --input INPUT
                        輸入影片、音訊檔案或包含媒體檔案的資料夾
  -d DURATION, --duration DURATION
                        每個片段的持續時間（秒）（預設值：900）
  -l LANG, --lang LANG  語言代碼（預設值：zh-TW）
  -m MODEL, --model MODEL
                        Gemini 模型（預設值：gemini-2.5-flash）
  --start START         開始時間（秒）
  --end END             結束時間（秒）
  --save-raw            儲存原始轉錄結果
  --skip-existing       如果 SRT 字幕檔案已存在則跳過處理
  --no-skip-existing    覆寫已存在的 SRT 檔案
  --debug               啟用 DEBUG 級別日誌記錄
  --max-workers MAX_WORKERS
                        最大工作執行緒數（預設值：依 CPU 與 API Key 計算）
  --extra-prompt EXTRA_PROMPT
                        額外的提示詞或包含提示詞的檔案路徑
  --ignore-keys-limit   忽略對最大工作執行緒數的 API Key 數量限制
  --preview             顯示原始轉錄結果預覽
  --max-segment-retries MAX_SEGMENT_RETRIES
                        每個音訊區塊最大重試次數
  --config CONFIG       設定檔路徑
```

### 💡 使用範例

1. **使用 TOML 配置 (推薦):**
   ```bash
   # 所有設定從 config.toml 讀取 - 僅需指定輸入檔案
   geminiasr -i video.mp4
   
   # 使用 TOML 設定處理整個目錄
   geminiasr -i /path/to/media/folder
   ```

2. **傳統命令列使用:**
   ```bash
   # 基本轉錄
   geminiasr -i video.mp4
   
   # 自訂設定
   geminiasr -i video.mp4 -d 300 --debug
   ```

## 🔍 音訊處理技術細節

> [!NOTE]
> 舊的預設模型 (`gemini-2.5-pro`) 是免費的但有一些限制。現在預設模型是 `gemini-2.5-flash`。

> [!IMPORTANT]
> 雖然 `gemini-3-pro-preview` 以及 `gemini-3-flash-preview` 已經推出，但在目前使用的 prompt template 下，對時間戳的判斷遠不如 `gemini-2.5-pro` 甚至是 `gemini-2.5-flash`，因此綜合考量還是推薦使用 `gemini-2.5-flash` 模型。

* 🧮 **Token 使用量**: Gemini 每秒音訊使用 32 個 token (1,920 tokens/分鐘)。有關音訊處理能力的更多詳細資訊，請參閱 [Gemini 音訊文件](https://ai.google.dev/gemini-api/docs/audio)。
* 📈 **輸出 Token**: Gemini 2.5 Pro/Flash 每個請求的輸出 token 限制為 65,536 個，這會影響可處理音訊的最大持續時間。有關詳細資訊，請參閱 [Gemini 模型文件](https://ai.google.dev/gemini-api/docs/models)。
* 📊 **速率限制**: 預設模型 (`gemini-2.5-pro`) 在預覽期間是免費的，但受特定限制：250,000 TPM (每分鐘 token)，5 RPM (每分鐘請求) 和 100 RPD (每天請求)。有關詳細資訊，請參閱 [速率限制文件](https://ai.google.dev/gemini-api/docs/rate-limits)。
* 💰 **定價**: 付費層每百萬 token 費用為 $1.25 (≤200k token) 或 $2.50 (>200k token)。對於超過 2 小時的音訊，建議分割檔案以避免過多的 token 使用量和潛在的成本超支。有關完整的定價資訊，請參閱 [Gemini 開發者 API 定價](https://ai.google.dev/gemini-api/docs/pricing)。

## 🤝 貢獻指南

感謝對 GeminiASR 的興趣！此指南讓貢獻流程更簡單一致。

### 快速開始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 開發備註

- 目標 Python：3.10+
- 設定：以 `config.example.toml` 為模板
- 避免提交機密：使用 `.env` 或未追蹤的 `config.toml`

### Lint 與測試

```bash
ruff check .
pytest
```

### PR 檢查清單

- [ ] 代碼已格式化並通過 lint
- [ ] 視需要新增或更新測試
- [ ] 若行為變更，已更新 README 或文件
- [ ] 未包含機密或憑證

#### Removed
- 舊版 `utils/` 模組

## 📄 授權

MIT License。詳見 `LICENSE`。

## 📝 注意事項

### 🔑 配置最佳實務
* **TOML 配置**：使用 `config.toml` 進行持久化設定。這是日常使用的推薦方式。
* **命令列覆蓋**：使用 CLI 參數暫時覆蓋 TOML 設定以進行特定運行。
* **多個 API Key**：在 TOML 或環境變數中配置多個 API Key 以獲得更好的效能。

### ⚡ 效能最佳化
* 為了獲得最佳效能，請考慮：
  * 🔑 使用多個 API Key (在 `config.toml` 或環境變數中配置)
  * ⏱️ 根據內容複雜性調整片段持續時間 — **將片段保持在 60 分鐘以下**，以避免輸出 token 限制，並維持免費層使用者的 TPM 限制
  * 🧵 根據系統能力和可用 API Key 的數量在 TOML 中適當配置 max-workers
  * 🚫 `ignore_keys_limit` 設定應謹慎使用，主要供付費層使用者使用，以避免觸及免費層嚴格的 TPM 限制

### 🚨 故障排除
* ⚠️ 如果您遇到 429 (請求過多) 錯誤，請嘗試在 config.toml 中減少 `max_workers` 設定，添加更多 API Key，或升級到付費層
* 💲 付費層使用者應使用 `gemini-2.5-pro` 模型，並且由於更高的 TPM 額度，可以安全地啟用 `ignore_keys_limit` 選項
* 🐛 在 config.toml 中使用 `debug = true` 或 `--debug` 標誌以查看詳細的配置和處理資訊

### 💰 成本管理
* 腳本使用 Gemini API，該 API 有使用限制，並可能在免費層之外產生費用。
* 啟用除錯模式時，透過詳細的配置日誌監控您的 token 使用量。
