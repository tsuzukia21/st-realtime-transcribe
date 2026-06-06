import logging
import re
import tempfile
import os
from flask import Flask, request
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# グローバル変数としてモデルを初期化
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

# 文の区切りとみなす日本語の句読点
SENTENCE_BOUNDARY = re.compile(r'(?<=[。？！、])')

def convert_seconds(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{int(minutes)}分{int(remaining_seconds)}秒"

def dedupe_sentences(text):
    """セグメント内で繰り返される文を、順序を保ったまま重複除去する。"""
    return ''.join(dict.fromkeys(SENTENCE_BOUNDARY.split(text)))


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        audio_file = request.files['audio']
        initial_prompt = request.form['initial_prompt']
        language = request.form['language']
    except KeyError as e:
        return f"missing field: {e}", 400

    # 一時ファイルに保存し、処理後に必ず削除する。
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        audio_file.save(tmp_path)

        segments, info = whisper_model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            task="transcribe",
            vad_filter=True,
            without_timestamps=True,
            initial_prompt=initial_prompt,
        )

        full_text_lines = []
        time_line_lines = []
        previous_text = ""
        for segment in segments:
            cleaned_text = dedupe_sentences(segment.text)
            if cleaned_text != previous_text:
                start = convert_seconds(segment.start)
                end = convert_seconds(segment.end)
                time_line_lines.append(f"[{start} -> {end}] {cleaned_text}  ")
                full_text_lines.append(cleaned_text)
            previous_text = cleaned_text

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "time_line": "\n".join(time_line_lines),
            "full_text": "\n".join(full_text_lines),
        }

    except Exception as e:
        logging.exception("transcription failed")
        return str(e), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
