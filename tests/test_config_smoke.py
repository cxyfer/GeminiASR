import os

from geminiasr.config import ConfigManager, DEFAULT_OPENAI_COMPAT_BASE_URL


def test_load_config_with_temp_toml(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("GEMINIASR_") or key in {"GOOGLE_API_KEY", "BASE_URL"}:
            monkeypatch.delenv(key, raising=False)

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[transcription]\n"
        "duration = 120\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    config = ConfigManager.load_config(str(config_path))

    assert config.transcription.duration == 120
    assert config.transcription.lang == "zh-TW"
    assert config.api.google_api_keys == []


def test_openai_source_defaults_base_url(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("GEMINIASR_") or key in {"GOOGLE_API_KEY", "BASE_URL"}:
            monkeypatch.delenv(key, raising=False)

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[api]\n"
        "source = \"openai\"\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    config = ConfigManager.load_config(str(config_path))

    assert config.api.source == "openai"
    assert config.api.base_url == DEFAULT_OPENAI_COMPAT_BASE_URL

