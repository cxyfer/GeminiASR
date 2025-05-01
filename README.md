# 🎙️ Gemini ASR Transcription Tool

A Python tool that uses Google Gemini API to transcribe video or audio files into SRT subtitle files. Supports multi-threading, file splitting, timestamp clipping, and custom prompts.

This tool leverages the powerful audio processing capabilities of Gemini 2.5 Pro, Google's advanced multimodal AI model, which excels at understanding and transcribing spoken content with high accuracy across multiple languages. 

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

## 🔧 Installation

### 🛠️ Environment Setup

1. **Install Python:** Python 3.8 or higher is recommended.
2. **Install uv:** If you haven't installed `uv` yet, please refer to the [uv official documentation](https://github.com/astral-sh/uv) for installation. `uv` is an extremely fast Python package installer and manager.

### 📦 Dependencies

Simply use `uv run` to automatically install and run the script:

```bash
uv run gemini_asr.py -i video.mp4
```

This will automatically install all required dependencies and execute the script.

### 🔑 API Keys

1. **Get Google API Key:** Go to [Google AI Studio](https://aistudio.google.com/app/apikey) to obtain your API key. You can get multiple keys to improve processing efficiency.
2. **Set Environment Variables:**
   * Create a file named `.env` in the project root directory.
   * Add your API key(s) to the `.env` file, separating multiple keys with commas:
     ```env
     GOOGLE_API_KEY=YOUR_API_KEY_1,YOUR_API_KEY_2,YOUR_API_KEY_3
     ```

## 📋 Usage

### ⌨️ Command Line Arguments

```
python gemini_asr.py [-h] -i INPUT [-d DURATION] [-l LANG] [-m MODEL]
                      [--start START] [--end END] [--save-raw]
                      [--skip-existing] [--debug]
                      [--max-workers MAX_WORKERS]
                      [--extra-prompt EXTRA_PROMPT]
                      [--ignore-keys-limit]

arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input video, audio file, or folder containing media files
  -d DURATION, --duration DURATION
                        Duration of each segment in seconds (default: 900)
  -l LANG, --lang LANG  Language code (default: zh-TW)
  -m MODEL, --model MODEL
                        Gemini model (default: gemini-2.5-pro-exp-03-25)
  --start START         Start time in seconds
  --end END             End time in seconds
  --save-raw            Save raw transcription results
  --skip-existing       Skip processing if SRT subtitle file already exists
  --debug               Enable DEBUG level logging
  --max-workers MAX_WORKERS
                        Maximum number of worker threads (default: cannot exceed the number of GOOGLE_API_KEYS)
  --extra-prompt EXTRA_PROMPT
                        Additional prompt or path to a file containing prompts
  --ignore-keys-limit   Ignore the API key quantity limit on maximum worker threads
```

### 💡 Examples

1. **Basic usage - transcribe a video file:**
   ```bash
   python gemini_asr.py -i video.mp4
   ```

2. **Transcribe a video with custom segment duration (5 minutes):**
   ```bash
   python gemini_asr.py -i video.mp4 -d 300
   ```

3. **Transcribe only a portion of the video (from 60s to 180s):**
   ```bash
   python gemini_asr.py -i video.mp4 --start 60 --end 180
   ```

4. **Process all media files in a directory:**
   ```bash
   python gemini_asr.py -i /path/to/media/folder
   ```

5. **Use a specific language:**
   ```bash
   python gemini_asr.py -i video.mp4 -l en
   ```

6. **Use custom prompt to improve transcription:**
   ```bash
   python gemini_asr.py -i lecture.mp4 --extra-prompt "This is a technical lecture about machine learning."
   ```

7. **Skip files that already have SRT subtitles:**
   ```bash
   python gemini_asr.py -i /path/to/media/folder --skip-existing
   ```

8. **Save raw transcription results for later review:**
   ```bash
   python gemini_asr.py -i interview.mp4 --save-raw
   ```

9. **Enable debug logging for troubleshooting:**
   ```bash
   python gemini_asr.py -i video.mp4 --debug
   ```

## 🔍 Technical Details About Audio Processing

* 🧮 **Token Usage**: Gemini uses 32 tokens per second of audio (1,920 tokens/minute). For more details on audio processing capabilities, see [Gemini Audio Documentation](https://ai.google.dev/gemini-api/docs/audio).
* 📈 **Output Tokens**: Gemini 2.5 Pro has a limit of 65,536 output tokens per request, which affects the maximum duration of processable audio. See [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-preview-03-25) for details.
* 📊 **Rate Limits**: The default model (`gemini-2.5-pro-exp-03-25`) is free during the preview period but subject to TPM (tokens per minute) limits. See [Rate Limits Documentation](https://ai.google.dev/gemini-api/docs/rate-limits) for details.
* 💰 **Pricing**: Paid tier costs $1.25 per million tokens (≤200k tokens) or $2.50 per million tokens (>200k tokens). See [Gemini Developer API Pricing](https://ai.google.dev/gemini-api/docs/pricing) for complete pricing information.

## 📝 Notes

* The script uses the Gemini API, which has usage limits and may incur costs beyond the free tier.
* For optimal performance, consider:
  * 🔑 Using multiple API keys (comma-separated in the .env file)
  * ⏱️ Adjusting the segment duration based on your content
  * 🧵 Setting appropriate max-workers based on your system capabilities
* ⚠️ If you encounter 429 (Too Many Requests) errors, try reducing the number of max-workers, adding more API keys, or upgrade to the paid tier and use the `--ignore-keys-limit` option.