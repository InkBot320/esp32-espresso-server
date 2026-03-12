from flask import Flask, request, jsonify
import requests
import struct
import os

app = Flask(__name__)
GROQ_KEY = os.environ.get("GROQ_API_KEY")

def build_wav(pcm_bytes, rate=8000):
    n = len(pcm_bytes)
    return struct.pack("<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + n, b"WAVE", b"fmt ", 16,
        1, 1, rate, rate * 2, 2, 16,
        b"data", n) + pcm_bytes

@app.route("/ping")
def ping():
    return "pong"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    pcm = request.data  # server holds all the bytes, not ESP32
    wav = build_wav(pcm)
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": "Bearer " + GROQ_KEY},
        files={"file": ("a.wav", wav, "audio/wav")},
        data={"model": "whisper-large-v3-turbo", "language": "en"}
    )
    text = r.json().get("text", "").strip()
    low = text.lower().strip(".,!? ")
    hallucinations = ["thank you", "thanks", "bye", "goodbye",
                      "you", "the", "thanks.", ""]
    if low in hallucinations:
        return jsonify({"text": ""})
    return jsonify({"text": text})

@app.route("/chat", methods=["POST"])
def chat():
    body = request.json
    time_str = body.get("time", "unknown")
    messages = body.get("messages", [])
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + GROQ_KEY,
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 150,
            "messages": [{"role": "system", "content":
                "You are a helpful voice assistant in Dublin Ireland. "
                "Current time is " + time_str + ". "
                "Be brief. Letters and numbers only, no special characters."
            }] + messages
        }
    )
    return jsonify({"reply": r.json()["choices"][0]["message"]["content"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
