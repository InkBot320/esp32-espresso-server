from flask import Flask, request, jsonify
import requests
import struct
import os
import threading
import time

app = Flask(__name__)
GROQ_KEY = os.environ.get("GROQ_API_KEY")

def build_wav_header(data_size, rate=8000):
    return struct.pack("<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size,
        b"WAVE", b"fmt ", 16,
        1, 1, rate, rate * 2, 2, 16,
        b"data", data_size)

def keep_alive():
    while True:
        time.sleep(270)  # ping itself every 4.5 minutes
        try:
            requests.get("https://esp32-espresso-server.onrender.com/ping", timeout=10)
            print("Keep alive ping sent")
        except:
            pass

# start keep alive thread on startup
t = threading.Thread(target=keep_alive, daemon=True)
t.start()

@app.route("/ping")
def ping():
    return "pong"

@app.route("/transcribe", methods=["POST"])
def transcribe():
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
        return "EMPTY", 200, {"Content-Type": "text/plain"}
    return text, 200, {"Content-Type": "text/plain"}

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
                "You are a helpful voice assistant. "
                "Current time is " + time_str + ". "
                "Be brief. Letters and numbers only, no special characters."
            }] + messages
        }
    )
    return jsonify({"reply": r.json()["choices"][0]["message"]["content"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
