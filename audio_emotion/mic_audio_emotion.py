import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from transformers import pipeline

# =========================
# CONFIG
# =========================
SAMPLE_RATE = 16000
DURATION = 10
AUDIO_FILE = "input.wav"
TEXT_CONF_THRESHOLD = 0.6

print("🎤 Recording started... Speak clearly")

# =========================
# STEP 1: RECORD AUDIO
# =========================
audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)
sd.wait()

audio = audio.flatten()

# Normalize audio safely
if np.max(np.abs(audio)) != 0:
    audio = audio / np.max(np.abs(audio))

write(AUDIO_FILE, SAMPLE_RATE, audio)
print("✅ Audio recorded and saved")

# =========================
# STEP 2: SPEECH → TEXT (WHISPER)
# =========================
print("\n📝 Transcribing speech...")
whisper_model = whisper.load_model("small")
transcription = whisper_model.transcribe(AUDIO_FILE)
text = transcription["text"].strip()

print("📝 Transcribed Text:", text)

# =========================
# STEP 3: TEXT → EMOTION
# =========================
text_emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

text_results = text_emotion_model(text)[0]
text_final = max(text_results, key=lambda x: x["score"])

text_emotion = text_final["label"]
text_score = text_final["score"]

# =========================
# STEP 4: AUDIO → EMOTION
# =========================
audio_emotion_model = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    framework="pt"
)

audio_results = audio_emotion_model(AUDIO_FILE)
audio_final = max(audio_results, key=lambda x: x["score"])

audio_map = {
    "hap": "joy",
    "ang": "anger",
    "sad": "sadness",
    "neu": "neutral"
}

audio_emotion = audio_map.get(audio_final["label"], audio_final["label"])
audio_score = audio_final["score"]

# =========================
# STEP 5: FUSION LOGIC
# =========================
if text_score >= TEXT_CONF_THRESHOLD:
    final_emotion = text_emotion
    reason = "Text emotion trusted (high confidence)"
else:
    final_emotion = audio_emotion
    reason = "Audio emotion used (text confidence low)"

# =========================
# STEP 6: EMOTION → SENTIMENT
# =========================
def emotion_to_sentiment(emotion):
    positive = ["joy", "happy"]
    negative = ["anger", "sadness", "fear", "disgust"]
    neutral = ["neutral"]

    emotion = emotion.lower()

    if emotion in positive:
        return "POSITIVE"
    elif emotion in negative:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

sentiment = emotion_to_sentiment(final_emotion)

# =========================
# FINAL OUTPUT
# =========================
print("\n🎧 Audio Emotion :", audio_emotion, f"({audio_score:.2f})")
print("📝 Text Emotion  :", text_emotion, f"({text_score:.2f})")

print("\n🔥 FINAL EMOTION :", final_emotion.upper())
print("📌 SENTIMENT    :", sentiment)
print("🧠 Reason       :", reason)
