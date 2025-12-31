TRANSCRIPTION_PROMPT_TEMPLATE = """請將這段音訊轉錄成文字，並包含時間戳。

每個時間戳的格式應為：[MM:SS.ss] 或 [HH:MM:SS.ss]，請務必包含小數點後的秒數，精確到小數點後兩位。
例如：
[00:01.25] 這是第一句話。（表示 1 秒 250 毫秒）
[00:05.78] 這是第二句話。（表示 5 秒 780 毫秒）
[01:23.45] 這是第三句話。（表示 1 分 23 秒 450 毫秒）

每個句子不要太長，以便用於字幕。每句話應該有明確的時間戳，反映說話的實際開始時間。
**重要規則：**
1. 每行字幕的文字內容（不包含時間戳）最多不應超過 50 個中文字。
2. 請移除每行文字末尾的標點符號（例如句號、逗號、問號、驚嘆號）。

如果有音樂或聲音效果，請標註如：
[01:02.35] [音樂] 或 [01:02.35] [音效]

請使用以下語言進行轉錄：{language}
"""


def get_transcription_prompt(lang: str, extra_prompt: str | None = None) -> str:
    prompt = TRANSCRIPTION_PROMPT_TEMPLATE.format(language=lang)
    if extra_prompt:
        prompt += f"\n\n此外，以下是一些額外的提示詞，請參考：\n{extra_prompt}"
    return prompt
