<div align="center">

# 🎙️ Gemini ASR Transcription Tool

![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

**English** | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

*A Python tool that uses Google Gemini API to transcribe video or audio files into SRT subtitle files.*

</div>

## ✨ Features

* 🎥 Supports various video (mp4, avi, mkv) and audio (mp3, wav) formats.
* ✂️ Automatically splits long files into smaller chunks for processing.
* 🧵 Uses multi-threading for parallel processing to speed up transcription.
* 🔄 Optionally rotates between multiple Google API Keys to improve request success rate.
* ⏱️ Generates SRT subtitle files with precise timestamps (millisecond accuracy).
* 🎬 Option to clip specific time segments of videos or audio for transcription.
* 📄 Option to save original transcription text returned by Gemini API.
* 💬 Supports passing additional prompts (text or file path) to guide the transcription model.
* 📁 Can process a single file or all supported files in an entire directory.
* ⏩ Provides `--skip-existing` option to avoid reprocessing files that already have subtitles.
* 🐞 Supports DEBUG mode for detailed logging output.
* 🌈 Uses colored logs for easy distinction between different message levels.
* 🔗 Supports proxy or custom server side like gemini-balance.
  * If you want to use gemini-balance, you need to set the `BASE_URL` environment variable to `https://your-custom-url.com/`.
  * Note: **code execution should be closed** if you use gemini-balance.
* ⚙️ **TOML Configuration Support**: Comprehensive configuration management system with multiple sources.
  * 📝 Configuration file support with automatic search in multiple locations
  * 🔄 Multi-source configuration merging (CLI > Environment > TOML > Defaults)
  * 🎛️ Easy management of all settings in a single configuration file

## 🔧 Installation

### 🛠️ Environment Setup

1. **Install Python:** Python 3.10 or higher is recommended.
2. **Install uv:** If you haven't installed `uv` yet, please refer to the [uv official documentation](https://github.com/astral-sh/uv) for installation. `uv` is an extremely fast Python package installer and manager.

### 📦 Install

**Option A: Editable install (recommended for development)**
```bash
pip install -e .
```

Then run:
```bash
geminiasr -i video.mp4
```

**Option B: One-shot run with uv**

```bash
uv run gemini_asr.py -i video.mp4
```

This will automatically install all required dependencies and execute the script.

### 🔑 API Keys Configuration

1. **Get Google API Key:** Go to [Google AI Studio](https://aistudio.google.com/app/apikey) to obtain your API key. You can get multiple keys to improve processing efficiency.

2. **Configuration Methods** (choose one):

   **Option A: TOML Configuration File (Recommended)**
   ```bash
   # Copy the example configuration file
   cp config.example.toml config.toml
   
   # Edit config.toml and add your API keys
   nano config.toml
   ```
   
   In `config.toml`:
   ```toml
   [api]
   google_api_keys = ["YOUR_API_KEY_1", "YOUR_API_KEY_2", "YOUR_API_KEY_3"]
   ```

   **Option B: Environment Variables**
   ```bash
   # Set environment variable with comma-separated keys
   export GOOGLE_API_KEY=YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3
   ```

   **Option C: .env File**
   ```bash
   # Create .env file in project root
   echo "GOOGLE_API_KEY=YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3" > .env
   ```

### ⚙️ Configuration System

GeminiASR supports a flexible configuration system with the following priority order:
1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **TOML configuration file**
4. **Default values** (lowest priority)

**Configuration File Locations** (searched in order):
- `./config.toml` (current directory)
- `./.geminiasr/config.toml`
- `~/.geminiasr/config.toml`
- `~/.config/geminiasr/config.toml`

**Environment Variable Whitelist**:
- `GOOGLE_API_KEY` (comma-separated keys)
- `GEMINIASR_LANG`, `GEMINIASR_MODEL`, `GEMINIASR_DURATION`
- `GEMINIASR_MAX_WORKERS`, `GEMINIASR_IGNORE_KEYS_LIMIT`, `GEMINIASR_DEBUG`
- `GEMINIASR_SAVE_RAW`, `GEMINIASR_SKIP_EXISTING`, `GEMINIASR_PREVIEW`
- `GEMINIASR_MAX_SEGMENT_RETRIES`, `GEMINIASR_EXTRA_PROMPT`
- `GEMINIASR_BASE_URL` or `BASE_URL`

**Example Configuration** (`config.toml`):
```toml
# Transcription Settings
[transcription]
duration = 900           # Segment duration in seconds
lang = "zh-TW"          # Language code
model = "gemini-2.5-flash"  # Gemini model
skip_existing = true     # Skip files with existing SRT
max_segment_retries = 3  # Max retries per segment

# Processing Settings
[processing]
max_workers = 24         # Max concurrent threads
ignore_keys_limit = true # Ignore API key limits

# Logging Settings
[logging]
debug = true            # Enable debug logging

# API Settings
[api]
google_api_keys = ["key1", "key2", "key3"]

# Advanced Settings
[advanced]
extra_prompt = "prompt.md"  # Path to prompt file
base_url = "https://generativelanguage.googleapis.com/"
```

## 📋 Usage

### ⌨️ Command Line Arguments

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
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input video, audio file, or folder containing media files
  -d DURATION, --duration DURATION
                        Duration of each segment in seconds (default: 900)
  -l LANG, --lang LANG  Language code (default: zh-TW)
  -m MODEL, --model MODEL
                        Gemini model (default: gemini-2.5-flash)
  --start START         Start time in seconds
  --end END             End time in seconds
  --save-raw            Save raw transcription results
  --skip-existing       Skip processing if SRT subtitle file already exists
  --no-skip-existing    Overwrite existing SRT files
  --debug               Enable DEBUG level logging
  --max-workers MAX_WORKERS
                        Maximum number of worker threads (default: based on CPU and API keys)
  --extra-prompt EXTRA_PROMPT
                        Additional prompt or path to a file containing prompts
  --ignore-keys-limit   Ignore the API key quantity limit on maximum worker threads
  --preview             Print a preview of raw transcription content
  --max-segment-retries MAX_SEGMENT_RETRIES
                        Max retries per segment
  --config CONFIG       Path to config.toml
```

### 💡 Usage Examples

1. **Using TOML Configuration (Recommended):**
   ```bash
   # All settings from config.toml - just specify input
   geminiasr -i video.mp4
   
   # Process entire directory with TOML settings
   geminiasr -i /path/to/media/folder
   ```

2. **Traditional Command-Line Usage:**
   ```bash
   # Basic transcription
   geminiasr -i video.mp4
   
   # With custom settings
   geminiasr -i video.mp4 -d 300 --debug
   ```

## 🔍 Technical Details About Audio Processing

> [!NOTE]
> The old default model (`gemini-2.5-pro`) is free but has some limits. Now the default model is `gemini-2.5-flash`.

* 🧮 **Token Usage**: Gemini uses 32 tokens per second of audio (1,920 tokens/minute). For more details on audio processing capabilities, see [Gemini Audio Documentation](https://ai.google.dev/gemini-api/docs/audio).
* 📈 **Output Tokens**: Gemini 2.5 Pro/Flash has a limit of 65,536 output tokens per request, which affects the maximum duration of processable audio. See [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models) for details.
* 📊 **Rate Limits**: The default model (`gemini-2.5-pro`) is free during the preview period but subject to specific limits: 250,000 TPM (tokens per minute), 5 RPM (requests per minute) and 100 RPD (requests per day). See [Rate Limits Documentation](https://ai.google.dev/gemini-api/docs/rate-limits) for details.
* 💰 **Pricing**: Paid tier costs $1.25 per million tokens (≤200k tokens) or $2.50 per million tokens (>200k tokens). For audio longer than 2 hours, it is recommended to split the file to avoid excessive token usage and potential cost overruns. See [Gemini Developer API Pricing](https://ai.google.dev/gemini-api/docs/pricing) for complete pricing information.

## 📄 License

MIT License. See `LICENSE`.

## 📝 Notes

### 🔑 Configuration Best Practices
* **TOML Configuration**: Use `config.toml` for persistent settings. This is the recommended approach for regular use.
* **Command-line Override**: Use CLI arguments to temporarily override TOML settings for specific runs.
* **Multiple API Keys**: Configure multiple API keys in TOML or environment variables for better performance.

### ⚡ Performance Optimization
* For optimal performance, consider:
  * 🔑 Using multiple API keys (configured in `config.toml` or environment variables)
  * ⏱️ Adjusting the segment duration based on content complexity—**keep segments under 60 minutes** to avoid output token limits and stay within TPM constraints for free tier users
  * 🧵 Configuring max-workers appropriately in TOML based on your system's capabilities and the number of available API keys
  * 🚫 The `ignore_keys_limit` setting should be used cautiously and primarily by paid tier users to avoid hitting the strict TPM limits of the free tier

### 🚨 Troubleshooting
* ⚠️ If you encounter 429 (Too Many Requests) errors, try reducing the `max_workers` setting in config.toml, adding more API keys, or upgrading to the paid tier
* 💲 Paid tier users should use the `gemini-2.5-pro` model and can safely enable the `ignore_keys_limit` option due to higher TPM allowances
* 🐛 Use `debug = true` in config.toml or `--debug` flag to see detailed configuration and processing information

### 💰 Cost Management
* The script uses the Gemini API, which has usage limits and may incur costs beyond the free tier.
* Monitor your token usage through the detailed configuration logging when debug mode is enabled.
