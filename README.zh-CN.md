<div align="center">

# 🎙️ Gemini ASR 语音转文字工具

[English](README.md) | **简体中文** | [繁體中文](README.zh-TW.md)

*一个使用 Google Gemini API 将视频或音频文件转录为 SRT 字幕文件的 Python 工具。*

</div>

---

# 🎙️ Gemini ASR 语音转文字工具

一个使用 Google Gemini API 将影片或音频文件转录为 SRT 字幕文件的 Python 工具。支持多线程、文件分割、时间戳剪辑和自定义提示词。

这个工具利用了 Google 先进的多模态 AI 模型 Gemini 2.5 Pro 强大的音频处理能力，它在理解和转录多种语言的口语内容方面表现出色，准确率很高。

## ✨ 功能

* 🎥 支持各种影片 (mp4, avi, mkv) 和音频 (mp3, wav) 格式。
* ✂️ 自动将长文件分割成较小的区块进行处理。
* 🧵 使用多线程进行并行处理以加速转录。
* 🔄 可选地轮换使用多个 Google API Key 以提高请求成功率。
* ⏱️ 生成具有精确时间戳 (毫秒精度) 的 SRT 字幕文件。
* 🎬 可选地剪辑影片或音频的特定时间段进行转录。
* 📄 可选地保存 Gemini API 返回的原始转录文字。
* 💬 支持传递额外的提示词 (文字或文件路径) 以引导转录模型。
* 📁 可以处理单一文件或整个目录中所有支持的文件。
* ⏩ 提供 `--skip-existing` 选项，以避免重新处理已经有字幕的文件。
* 🐞 支持 DEBUG 模式，用于详细的日志输出。
* 🌈 使用彩色日志，方便区分不同级别的訊息。

## 🔧 安装

### 🛠️ 环境设定

1. **安装 Python:** 建议使用 Python 3.8 或更高版本。
2. **安装 uv:** 如果您尚未安装 `uv`，请参阅 [uv 官方文件](https://github.com/astral-sh/uv) 进行安装。`uv` 是一个极快的 Python 包安装和管理器。

### 📦 依赖

只需使用 `uv run` 即可自动安装和运行脚本：

```bash
uv run gemini_asr.py -i video.mp4
```

这将自动安装所有所需的依赖并执行脚本。

### 🔑 API Key

1. **获取 Google API Key:** 前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 获取您的 API Key。
您可以获取多个 Key 以提高处理效率。
2. **设定环境变量:**
   * 在项目根目录中创建一个名为 `.env` 的文件。
   * 在 `.env` 文件中加入您的 API Key(s)，多个 Key 使用逗号分隔：
     ```env
     GOOGLE_API_KEY=YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3
     ```

## 📋 使用方法

### ⌨️ 命令行参数

```
python gemini_asr.py [-h] -i INPUT [-d DURATION] [-l LANG] [-m MODEL]
                      [--start START] [--end END] [--save-raw]
                      [--skip-existing] [--debug]
                      [--max-workers MAX_WORKERS]
                      [--extra-prompt EXTRA_PROMPT]
                      [--ignore-keys-limit]

arguments:
  -h, --help            显示帮助訊息并退出
  -i INPUT, --input INPUT
                        输入影片、音频文件或包含媒体文件的文件夹
  -d DURATION, --duration DURATION
                        每个片段的持续时间（秒）（默认值：900）
  -l LANG, --lang LANG  语言代码（默认值：zh-TW）
  -m MODEL, --model MODEL
                        Gemini 模型（默认值：gemini-2.5-pro-exp-03-25）
  --start START         开始时间（秒）
  --end END             结束时间（秒）
  --save-raw            保存原始转录结果
  --skip-existing       如果 SRT 字幕文件已存在则跳过处理
  --debug               启用 DEBUG 级别日志记录
  --max-workers MAX_WORKERS
                        最大工作线程数（默认值：不能超过 GOOGLE_API_KEY 的数量）
  --extra-prompt EXTRA_PROMPT
                        额外的提示词或包含提示词的文件路径
  --ignore-keys-limit   忽略对最大工作线程数的 API Key 数量限制
```

### 💡 示例

1. **基本使用 - 转录影片文件:**
   ```bash
   python gemini_asr.py -i video.mp4
   ```

2. **转录影片并使用自定义片段持续时间 (5 分钟):**
   ```bash
   python gemini_asr.py -i video.mp4 -d 300
   ```

3. **仅转录影片的某一部分 (从 60 秒到 180 秒):**
   ```bash
   python gemini_asr.py -i video.mp4 --start 60 --end 180
   ```

4. **处理目录中的所有媒体文件:**
   ```bash
   python gemini_asr.py -i /path/to/media/folder
   ```

5. **使用特定语言:**
   ```bash
   python gemini_asr.py -i video.mp4 -l en
   ```

6. **使用自定义提示词改善转录:**
   ```bash
   python gemini_asr.py -i lecture.mp4 --extra-prompt "This is a technical lecture about machine learning."
   ```

7. **跳过已经有 SRT 字幕的文件:**
   ```bash
   python gemini_asr.py -i /path/to/media/folder --skip-existing
   ```

8. **保存原始转录结果以供后续审查:**
   ```bash
   python gemini_asr.py -i interview.mp4 --save-raw
   ```

9. **启用调试日志记录以进行故障排除:**
   ```bash
   python gemini_asr.py -i video.mp4 --debug
   ```

## 🔍 音频处理技术细节

* 🧮 **Token 使用量**: Gemini 每秒音频使用 32 个 token (1,920 tokens/分钟)。有关音频处理能力的更多详细信息，请参阅 [Gemini 音频文件](https://ai.google.dev/gemini-api/docs/audio)。
* 📈 **输出 Token**: Gemini 2.5 Pro 每个请求的输出 token 限制为 65,536 个，这会影响可处理音频的最大持续时间。
有关详细信息，请参阅 [Gemini 模型文件](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-preview-03-25)。
* 📊 **速率限制**: 默认模型 (`gemini-2.5-pro-exp-03-25`) 在预览期间是免费的，但受特定限制：250,000 TPM (每分钟 token)，1,000,000 TPD (每天 token)，5 RPM (每分钟请求) 和 25 RPM (每分钟请求)。有关详细信息，请参阅 [速率限制文件](https://ai.google.dev/gemini-api/docs/rate-limits)。
* 💰 **定价**: 付费层每百万 token 费用为 $1.25 (≤200k token) 或 $2.50 (>200k token)。对于超过 2 小时的音频，建议分割文件以避免过多的 token 使用量和潜在的成本超支。有关完整的定价信息，请参阅 [Gemini 开发者 API 定价](https://ai.google.dev/gemini-api/docs/pricing)。

## 📝 注意事项

* 脚本使用 Gemini API，该 API 有使用限制，并可能在免费层之外产生费用。
* 为了获得最佳性能，请考虑：
  * 🔑 使用多个 API Key (在 .env 文件中使用逗号分隔)
  * ⏱️ 根据内容复杂性调整片段持续时间 — **将片段保持在 60 分钟以下**，以避免输出 token 限制，并维持免费层使用者的 TPM 限制
  * 🧵 根据系统能力和可用 API Key 的数量适当配置 max-workers
  * 🚫 `--ignore-keys-limit` 选项应谨慎使用，主要供付费层使用者使用，以避免触及免费层严格的 TPM 限制
* ⚠️ 如果您遇到 429 (请求过多) 错误，请尝试减少 max-workers 的数量，添加更多 API Key，或升级到付费层
* 💲 付费层使用者应使用 `gemini-2.5-pro-preview-03-25` 模型，并且由于更高的 TPM 额度，可以安全地使用 `--ignore-keys-limit` 选项。

---
