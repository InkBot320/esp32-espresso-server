from flask import Flask, request, jsonify
import requests
import struct
import os

app = Flask(__name__)
GROQ_KEY = os.environ.get("GROQ_API_KEY")

def build_wav_header(data_size, rate=8000):
    return struct.pack("<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size,
        b"WAVE", b"fmt ", 16,
        1, 1, rate, rate * 2, 2, 16,
        b"data", data_size)

@app.route("/ping")
def ping():
    return "pong"

@app.route("/transcribe", methods=["POST"])
def transcribe():
    # receive raw pcm, build wav, send to whisper
    pcm = request.data
    print("Received", len(pcm), "bytes")
    data_size = len(pcm)
    wav = build_wav_header(data_size) + pcm
    r = requests.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": "Bearer " + GROQ_KEY},
        files={"file": ("a.wav", wav, "audio/wav")},
        data={"model": "whisper-large-v3-turbo", "language": "en"}
    )
    result = r.json()
    print("Whisper:", result)
    text = result.get("text", "").strip()
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
