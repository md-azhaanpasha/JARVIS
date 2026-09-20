import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from groq import Groq

app = FastAPI()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are Jarvis, a helpful AI assistant embedded in a small ESP32 device "
    "with a tiny OLED screen. Keep every reply SHORT — 1 to 3 sentences, "
    "plain text, no markdown, no emojis. Be direct and useful."
)

# Very simple in-memory conversation history (resets on server restart).
# Fine for a single-user hobby device.
conversation_history = []
MAX_HISTORY_TURNS = 6  # keep last N exchanges to limit token usage


@app.get("/")
def health_check():
    return {"status": "Jarvis server is running"}


@app.post("/jarvis")
async def jarvis_command(audio: UploadFile = File(...)):
    global conversation_history

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio received")

    # --- 1. Transcribe with Groq Whisper ---
    try:
        transcription = client.audio.transcriptions.create(
            file=(audio.filename or "audio.wav", io.BytesIO(audio_bytes)),
            model="whisper-large-v3-turbo",
        )
        transcript = transcription.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    if not transcript:
        return JSONResponse({"transcript": "", "reply": "I didn't catch that."})

    # --- 2. Get a reply from Groq's Llama model ---
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": transcript})

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=150,
            temperature=0.6,
        )
        reply = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {e}")

    # --- 3. Update short conversation history ---
    conversation_history.append({"role": "user", "content": transcript})
    conversation_history.append({"role": "assistant", "content": reply})
    conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]

    return JSONResponse({"transcript": transcript, "reply": reply})
