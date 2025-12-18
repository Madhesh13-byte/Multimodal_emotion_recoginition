# 🎧 Audio Emotion Detection using wav2vec2

This project implements **speech emotion recognition** using a **pretrained wav2vec2 model** fine-tuned on the **SUPERB benchmark**, powered by **Hugging Face Transformers**.  
The system analyzes human speech and predicts emotional states such as **Happy, Angry, Neutral, and Sad**.

---

## 🔥 Tech Stack
- Python 3.10
- Hugging Face Transformers
- wav2vec2 (SUPERB Emotion Model)
- PyTorch
- ffmpeg

---

## 🧠 Model Details
- **Model Name:** `superb/wav2vec2-base-superb-er`
- **Architecture:** CNN + Transformer (wav2vec2)
- **Task:** Speech Emotion Recognition (SER)
- **Input:** WAV audio (16 kHz, mono)
- **Output:** Emotion probabilities

---

## ▶️ How to Run

1️⃣ Install dependencies:
```bash
pip install transformers torch torchaudio soundfile

2️⃣ Ensure ffmpeg is installed:

ffmpeg -version


3️⃣ Run emotion detection:

python test_audio_emotion.py

📊 Sample Output
hap : 0.664
ang : 0.222
neu : 0.084
sad : 0.030