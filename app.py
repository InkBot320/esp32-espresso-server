from flask import Flask, request, jsonify
import requests
import struct
import os
import threading
import time
import io

app = Flask(__name__)
GROQ_KEY = os.environ.get("GROQ_API_KEY")
VOICERSS_KEY = os.environ.get("VOICERSS_API_KEY")

def build_wav_header(data_size, rate=12000):
    return struct.pack("<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size,
        b"WAVE", b"fmt ", 16,
        1, 1, rate, rate * 2, 2, 16,
        b"data", data_size)

def keep_alive():
    while True:
        time.sleep(270)
        try:
            requests.get("https://esp32-espresso-server.onrender.com/ping", timeout=10)
            print("Keep alive ping sent")
        except:
            pass

t = threading.Thread(target=keep_alive, daemon=True)
t.start()

@app.route("/ping")
def ping():
    return "pong"

@app.route("/")
def home():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Espresso</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #1a1a1a; color: #fff; font-family: sans-serif; height: 100vh; display: flex; flex-direction: column; }
        #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 10px; }
        .msg { max-width: 70%; padding: 10px 15px; border-radius: 18px; line-height: 1.4; }
        .user { background: #6f4e37; align-self: flex-end; }
        .bot { background: #2a2a2a; align-self: flex-start; }
        .bot-name { color: #c8956c; font-size: 11px; margin-bottom: 4px; }
        #input-area { padding: 15px; display: flex; gap: 10px; background: #111; }
        #input { flex: 1; padding: 12px; border-radius: 25px; border: none; background: #2a2a2a; color: #fff; font-size: 16px; outline: none; }
        #send { padding: 12px 20px; border-radius: 25px; border: none; background: #6f4e37; color: #fff; cursor: pointer; font-size: 16px; }
        #send:hover { background: #8b6347; }
        .thinking { color: #888; font-style: italic; }
    </style>
</head>
<body>
    <div id="chat">
        <div class="msg bot">
            <div class="bot-name">Espresso</div>
            Hey! Ask me anything.
        </div>
    </div>
    <div id="input-area">
        <input id="input" type="text" placeholder="Ask Espresso anything..." autofocus/>
        <button id="send" onclick="sendMsg()">Send</button>
    </div>
    <script>
        let messages = [];
        const chat = document.getElementById("chat");
        const input = document.getElementById("input");

        input.addEventListener("keydown", e => {
            if (e.key === "Enter") sendMsg();
        });

        function addMsg(text, role) {
            const div = document.createElement("div");
            div.className = "msg " + (role === "user" ? "user" : "bot");
            if (role === "bot") {
                const name = document.createElement("div");
                name.className = "bot-name";
                name.textContent = "Espresso";
                div.appendChild(name);
            }
            div.appendChild(document.createTextNode(text));
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        async function sendMsg() {
            const text = input.value.trim();
            if (!text) return;
            input.value = "";
            addMsg(text, "user");
            messages.push({ role: "user", content: text });

            const thinking = addMsg("thinking...", "bot");
            thinking.classList.add("thinking");

            try {
                const r = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ time: new Date().toLocaleString(), messages })
                });
                const data = await r.json();
                thinking.remove();
                const reply = data.reply || "No response";
                addMsg(reply, "bot");
                messages.push({ role: "assistant", content: reply });
            } catch(e) {
                thinking.textContent = "Error, try again.";
            }
        }
    </script>
</body>
</html>
'''

@app.route("/transcribe", methods=["POST"])
def transcribe():
    pcm = request.data
    print("Received", len(pcm), "bytes")
    wav = build_wav_header(len(pcm)) + pcm
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
            "max_tokens": 60,
            "messages": [{"role": "system", "content":
                "You are a helpful voice assistant called Espresso in Dublin Ireland. "
                "Current time is " + time_str + ". "
                "Be helpful and conversational. Keep answers concise but complete."
            }] + messages
        }
    )
    return jsonify({"reply": r.json()["choices"][0]["message"]["content"]})

@app.route("/tts", methods=["POST"])
def tts():
    text = request.json.get("text", "")
    r = requests.get(
        "https://api.voicerss.org/",
        params={
            "key": VOICERSS_KEY,
            "hl": "en-ie",
            "v": "Mary",
            "src": text,
            "c": "WAV",
            "f": "22khz_16bit_mono"
        }
    )
    print("TTS bytes:", len(r.content), "first:", r.content[:4])
    return r.content[44:], 200, {"Content-Type": "application/octet-stream"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
