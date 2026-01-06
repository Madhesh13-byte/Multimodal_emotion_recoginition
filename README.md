# HSEmotion MediaPipe System

Real-time emotion recognition using MediaPipe face detection and HSEmotion VA analysis.

## Installation

1.  **Prerequisites**: Python 3.8 or higher.

2.  **Install Dependencies**:
    ```bash
    pip install hsemotion timm opencv-python mediapipe numpy onnxruntime
    ```
    *Note: If you have issues with `timm`, this system patches compatibility automatically.*

3.  **Download Models**:
    The system requires the `blaze_face_short_range.tflite` model for face detection.
    
    Powershell command to download:
    ```powershell
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" -OutFile "blaze_face_short_range.tflite"
    ```

## Usage

Run the main system:
```bash
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