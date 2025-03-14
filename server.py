from flask import Flask, request, jsonify
import logging
logging.basicConfig(level=logging.DEBUG)
import re
app = Flask(__name__)
from faster_whisper import WhisperModel

# グローバル変数としてモデルを初期化
# whisper_model = WhisperModel("turbo", device="cuda", compute_type="float16")
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def convert_seconds(seconds):
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{int(minutes)}分{int(remaining_seconds)}秒"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    model = whisper_model
    try:
        audio_file = request.files['audio']
        file_name = request.form['file_name']
        initial_prompt = request.form['initial_prompt']
        language = request.form['language']
        audio_file.save(file_name)
        
        segments, info = model.transcribe(file_name,
                                          language = language,
                                            beam_size = 5,
                                            task = "transcribe",
                                            vad_filter=True,
                                            without_timestamps = True,
                                            initial_prompt = initial_prompt
                                            )
        full_text=""
        time_line=""
        segment_before=""
        for segment in segments:
            sentences = re.split('(?<=[。？！、])', segment.text)
            seen_sentences = set()
            cleaned_sentences = []
            for sentence in sentences:
                if sentence not in seen_sentences:
                    cleaned_sentences.append(sentence)
                    seen_sentences.add(sentence)
            cleaned_text = ''.join(cleaned_sentences)
            if not cleaned_text==segment_before:
                time_line+="[%s -> %s] %s" % (convert_seconds(segment.start), convert_seconds(segment.end), cleaned_text)+"  \n"
                full_text+=cleaned_text+"\n"
            segment_before=cleaned_text

        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "time_line":time_line,
            "full_text":full_text
        }

        return result

    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=XXXX, debug=False)