import io
import torch
import torchaudio
import base64
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, send_file, render_template, jsonify
from audiocraft.models import AudioGen
from translate_model import run_prediction  # your custom module

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AudioGen.get_pretrained("facebook/audiogen-medium")
model.set_generation_params(duration=4, top_k=250, temperature=1.0, cfg_coef=3.0)
torch.manual_seed(42)

last_audio_buffer = None


def convert_to_humming_prompt(text: str) -> str:
    return (
        f"A human-like, expressive humming mimics the natural rhythm, intonation, and emotion of the sentence '{text}' at a slower, more deliberate tempo, "
        f"without using any actual speech or words. It is as if a person is communicating nonverbally using a series of drawn-out grunts and extended hums, "
        f"where grunts represent consonants such as 't', 'k', and 'm', and hums represent vowels like 'ah', 'ee', and 'oo'. "
        f"This pattern emulates human speech structure using measured pitch and tone only."
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    global last_audio_buffer
    sentence = request.form.get("sentence")
    if not sentence:
        return "No sentence provided", 400

    prompt = convert_to_humming_prompt(sentence)
    with torch.inference_mode():
        wav = model.generate([prompt])[0].detach().cpu()

    buffer = io.BytesIO()
    torchaudio.save(buffer, wav, sample_rate=model.sample_rate, format="wav")
    buffer.seek(0)
    last_audio_buffer = buffer.getvalue()

    return send_file(
        io.BytesIO(last_audio_buffer),
        mimetype="audio/wav",
        as_attachment=False,
        download_name="output.wav",
    )


@app.route("/upload", methods=["POST"])
def upload():
    global last_audio_buffer
    file = request.files.get("audio")
    if file is None:
        return "No file uploaded", 400
    buffer = io.BytesIO(file.read())
    buffer.seek(0)
    last_audio_buffer = buffer.getvalue()

    return send_file(
        io.BytesIO(last_audio_buffer),
        mimetype="audio/wav",
        as_attachment=False,
        download_name="uploaded.wav",
    )


@app.route("/translate", methods=["POST"])
def translate():
    if last_audio_buffer is None:
        return jsonify({"output": "No audio has been generated yet."}), 400
    try:
        output = run_prediction(last_audio_buffer)
        return jsonify({"output": output})
    except Exception as e:
        return jsonify({"output": f"Error during translation: {str(e)}"}), 500


@app.route("/analyze-pitch", methods=["POST"])
def analyze_pitch():
    if last_audio_buffer is None:
        return jsonify({"error": "No audio has been generated yet."}), 400
    y, sr = librosa.load(io.BytesIO(last_audio_buffer), sr=16000)
    f0 = librosa.yin(y, fmin=80, fmax=500, sr=sr)
    times = librosa.times_like(f0)
    f0[f0 == 0] = np.nan

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, f0, label="Estimated Pitch (Hz)", color="cyan")
    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_ylabel("Pitch (Hz)", fontsize=16)
    ax.set_title("Humming Pitch Contour", fontsize=18)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return jsonify({"image": image_base64})


if __name__ == "__main__":
    app.run(debug=True)
