from transformers import pipeline

# Load SUPERB emotion recognition model
emotion_classifier = pipeline(
    task="audio-classification",
    model="superb/wav2vec2-base-superb-er"
)

# Run prediction
result = emotion_classifier("harvard.wav")

print("Emotion predictions:")
for r in result:
    print(r["label"], ":", round(r["score"], 3))
