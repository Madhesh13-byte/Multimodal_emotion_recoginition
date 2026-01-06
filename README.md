# HSEmotion MediaPipe System

Real-time emotion recognition using MediaPipe face detection and HSEmotion VA analysis.

## Quick Start

```bash
pip install -r requirements.txt
python mediapipe_system.py
```

## Features

- **MediaPipe Face Detection**: Superior accuracy, no Haar cascades
- **HSEmotion VA Analysis**: Continuous valence/arousal prediction
- **Russell's Circumplex**: VA-based emotion classification
- **Real-time Processing**: 2 analyses per second
- **JSON Logging**: Timestamped emotion data

## Output

- **Valence**: [-1, +1] emotional polarity
- **Arousal**: [0, +1] emotional intensity
- **Emotions**: Happy, Sad, Anger, Surprise, Disgust, Neutral, etc.
- **Confidence**: Classification confidence scores

## Controls

- Press 'q' to quit
- Logs saved as `mediapipe_emotion_log.json`