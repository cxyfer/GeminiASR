<div align="center">

# 🎙️ Gemini ASR 语音转文字工具

[English](README.md) | **简体中文** | [繁體中文](README.zh-TW.md)

*一个使用 Google Gemini API 将视频或音频文件转录为 SRT 字幕文件的 Python 工具。*

</div>


## ✨ 功能

* 🎥 支持各种视频 (mp4, avi, mkv) 和音频 (mp3, wav) 格式。
* ✂️ 自动将长文件分割成较小的区块进行处理。
* 🧵 使用多线程进行并行处理以加速转录。
* 🔄 可选地轮换使用多个 Google API Key 以提高请求成功率。
* ⏱️ 生成具有精确时间戳 (毫秒精度) 的 SRT 字幕文件。
* 🎬 可选地剪辑视频或音频的特定时间段进行转录。
* 📄 可选地保存 Gemini API 返回的原始转录文字。
* 💬 支持传递额外的提示词 (文字或文件路径) 以引导转录模型。
* 📁 可以处理单一文件或整个目录中所有支持的文件。
* ⏩ 提供 `--skip-existing` 选项，以避免重新处理已经有字幕的文件。
* 🐞 支持 DEBUG 模式，用于详细的日志输出。
* 🌈 使用彩色日志，方便区分不同级别的消息。
* 🔗 支持代理或自定义服务器端，如 gemini-balance。
  * 如果要使用 gemini-balance，需要将 `BASE_URL` 环境变量设置为 `https://your-custom-url.com/`。
  * 注意：如果使用 gemini-balance，需要**关闭代码执行**。
* ⚙️ **TOML 配置支持**：全面的配置管理系统，支持多种配置来源。
  * 📝 支持配置文件，在多个位置自动搜索
  * 🔄 多源配置合并 (命令行 > 环境变量 > TOML > 默认值)
  * 🎛️ 在单个配置文件中轻松管理所有设置

## 🔧 安装

### 🛠️ 环境设定

1. **安装 Python:** 建议使用 Python 3.10 或更高版本。
2. **安装 uv:** 如果您尚未安装 `uv`，请参阅 [uv 官方文件](https://github.com/astral-sh/uv) 进行安装。`uv` 是一个极快的 Python 包安装和管理器。

### 📦 安装

**选项 A：可编辑安装（推荐开发使用）**
```bash
pip install -e .
```

然后运行：
```bash
geminiasr -i video.mp4
```

**选项 B：使用 uv 一键运行**

```bash
uv run gemini_asr.py -i video.mp4
```

这将自动安装所有所需的依赖并执行脚本。

### 🔑 API Keys 配置

1. **获取 Google API Key:** 前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 获取您的 API Key。您可以获取多个 Key 以提高处理效率。

2. **配置方法** (选择其中一种):

   **方式 A: TOML 配置文件 (推荐)**
   ```bash
   # 复制示例配置文件
   cp config.example.toml config.toml
   
   # 编辑 config.toml 并添加您的 API Key
   nano config.toml
   ```
   
   在 `config.toml` 中:
   ```toml
   [api]
   source = "gemini"  # "gemini" 或 "openai"
   google_api_keys = ["YOUR_API_KEY_1", "YOUR_API_KEY_2", "YOUR_API_KEY_3"]
   ```

   **方式 B: 环境变量**
   ```bash
   # 设置环境变量，多个 Key 用逗号分隔
   export GOOGLE_API_KEY=YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3
   ```

   **方式 C: .env 文件**
   ```bash
   # 在项目根目录创建 .env 文件
   echo "GOOGLE_API_KEY=YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3" > .env
   ```

### ⚙️ 配置系统

GeminiASR 支持灵活的配置系统，优先级顺序如下:
1. **命令行参数** (最高优先级)
2. **环境变量**
3. **TOML 配置文件**
4. **默认值** (最低优先级)

**配置文件搜索位置** (按顺序搜索):
- `./config.toml` (当前目录)
- `./.geminiasr/config.toml`
- `~/.geminiasr/config.toml`
- `~/.config/geminiasr/config.toml`

**环境变量白名单**:
- `GOOGLE_API_KEY` (逗号分隔)
- `GEMINIASR_LANG`, `GEMINIASR_MODEL`, `GEMINIASR_DURATION`
- `GEMINIASR_MAX_WORKERS`, `GEMINIASR_IGNORE_KEYS_LIMIT`, `GEMINIASR_DEBUG`
- `GEMINIASR_SAVE_RAW`, `GEMINIASR_SKIP_EXISTING`, `GEMINIASR_PREVIEW`
- `GEMINIASR_MAX_SEGMENT_RETRIES`, `GEMINIASR_EXTRA_PROMPT`
- `GEMINIASR_API_SOURCE`
- `GEMINIASR_BASE_URL` 或 `BASE_URL`

**OpenAI 兼容端点**:
- 设置 `api.source = "openai"`（或 `GEMINIASR_API_SOURCE=openai`）。
- 若 `advanced.base_url` 保持 Gemini 默认值，会自动切换为 `https://generativelanguage.googleapis.com/v1beta/openai/`。

**配置示例** (`config.toml`):
```toml
# 转录设置
[transcription]
duration = 900           # 片段持续时间（秒）
lang = "zh-TW"          # 语言代码
model = "gemini-2.5-flash"  # Gemini 模型
skip_existing = true     # 跳过已有 SRT 文件
max_segment_retries = 3  # 每个片段最大重试次数

# 处理设置
[processing]
max_workers = 24         # 最大并发线程数
ignore_keys_limit = true # 忽略 API Key 限制

# 日志设置
[logging]
debug = true            # 启用调试日志

# API 设置
[api]
source = "gemini"  # "gemini" 或 "openai"
google_api_keys = ["key1", "key2", "key3"]

# 高级设置
[advanced]
extra_prompt = "prompt.md"  # 提示词文件路径
base_url = "https://generativelanguage.googleapis.com/"
# base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
```

## 📋 使用方法

### ⌨️ 命令行参数

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
  -h, --help            显示帮助訊息并退出
  -i INPUT, --input INPUT
                        输入影片、音频文件或包含媒体文件的文件夹
  -d DURATION, --duration DURATION
                        每个片段的持续时间（秒）（默认值：900）
  -l LANG, --lang LANG  语言代码（默认值：zh-TW）
  -m MODEL, --model MODEL
                        Gemini 模型（默认值：gemini-2.5-flash）
  --start START         开始时间（秒）
  --end END             结束时间（秒）
  --save-raw            保存原始转录结果
  --skip-existing       如果 SRT 字幕文件已存在则跳过处理
  --no-skip-existing    覆盖已存在的 SRT 文件
  --debug               启用 DEBUG 级别日志记录
  --max-workers MAX_WORKERS
                        最大工作线程数（默认值：基于 CPU 与 API Key）
  --extra-prompt EXTRA_PROMPT
                        额外的提示词或包含提示词的文件路径
  --ignore-keys-limit   忽略对最大工作线程数的 API Key 数量限制
  --preview             显示原始转录结果预览
  --max-segment-retries MAX_SEGMENT_RETRIES
                        每个音频片段最大重试次数
  --config CONFIG       配置文件路径
```

### 💡 使用示例

1. **使用 TOML 配置 (推荐):**
   ```bash
   # 所有设置从 config.toml 读取 - 只需指定输入文件
   geminiasr -i video.mp4
   
   # 使用 TOML 设置处理整个目录
   geminiasr -i /path/to/media/folder
   ```

2. **传统命令行用法:**
   ```bash
   # 基本转录
   geminiasr -i video.mp4
   
   # 自定义设置
   geminiasr -i video.mp4 -d 300 --debug
   ```

## 🔍 音频处理技术细节

> [!NOTE]
> 旧的默认模型 (`gemini-2.5-pro`) 是免费的但有一些限制。现在默认模型是 `gemini-2.5-flash`。

* 🧮 **Token 使用量**: Gemini 每秒音频使用 32 个 token (1,920 tokens/分钟)。有关音频处理能力的更多详细信息，请参阅 [Gemini 音频文档](https://ai.google.dev/gemini-api/docs/audio)。
* 📈 **输出 Token**: Gemini 2.5 Pro/Flash 每个请求的输出 token 限制为 65,536 个，这会影响可处理音频的最大持续时间。详情请参阅 [Gemini 模型文档](https://ai.google.dev/gemini-api/docs/models)。
* 📊 **速率限制**: 默认模型 (`gemini-2.5-pro`) 在预览期间是免费的，但受特定限制：250,000 TPM (每分钟 token)，5 RPM (每分钟请求) 和 100 RPD (每日请求)。详情请参阅 [速率限制文档](https://ai.google.dev/gemini-api/docs/rate-limits)。
* 💰 **定价**: 付费层每百万 token 费用为 $1.25 (≤200k token) 或 $2.50 (>200k token)。对于超过 2 小时的音频，建议分割文件以避免过多的 token 使用量和潜在的成本超支。有关完整的定价信息，请参阅 [Gemini 开发者 API 定价](https://ai.google.dev/gemini-api/docs/pricing)。

## 🤝 贡献指南

感谢你对 GeminiASR 的兴趣！此指南让贡献流程更简单一致。

### 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 开发备注

- 目标 Python：3.10+
- 配置：以 `config.example.toml` 为模板
- 避免提交机密：使用 `.env` 或未追踪的 `config.toml`

### Lint 与测试

```bash
ruff check .
pytest
```

### PR 检查清单

- [ ] 代码已格式化并通过 lint
- [ ] 视需要新增或更新测试
- [ ] 若行为变更，已更新 README 或文档
- [ ] 未包含机密或凭证

## 📄 许可证

MIT License。详见 `LICENSE`。

## 📝 注意事项

### 🔑 配置最佳实践
* **TOML 配置**: 使用 `config.toml` 进行持久化设置。这是日常使用的推荐方法。
* **命令行覆盖**: 使用 CLI 参数临时覆盖 TOML 设置以进行特定运行。
* **多个 API Key**: 在 TOML 或环境变量中配置多个 API Key 以获得更好的性能。

### ⚡ 性能优化
* 为了获得最佳性能，请考虑：
  * 🔑 使用多个 API Key (在 `config.toml` 或环境变量中配置)
  * ⏱️ 根据内容复杂性调整片段持续时间 — **将片段保持在 60 分钟以下**，以避免输出 token 限制，并维持免费层用户的 TPM 限制
  * 🧵 根据系统能力和可用 API Key 的数量在 TOML 中适当配置 max-workers
  * 🚫 `ignore_keys_limit` 设置应谨慎使用，主要供付费层用户使用，以避免触及免费层严格的 TPM 限制

### 🚨 故障排除
* ⚠️ 如果您遇到 429 (请求过多) 错误，请尝试减少 config.toml 中的 `max_workers` 设置，添加更多 API Key，或升级到付费层
* 💲 付费层用户应使用 `gemini-2.5-pro` 模型，并且由于更高的 TPM 限额，可以安全地启用 `ignore_keys_limit` 选项
* 🐛 在 config.toml 中使用 `debug = true` 或 `--debug` 标志查看详细的配置和处理信息

### 💰 成本管理
* 脚本使用 Gemini API，该 API 有使用限制，并可能在免费层之外产生费用。
* 启用调试模式时，通过详细的配置日志监控您的 token 使用情况。
