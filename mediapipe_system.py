import cv2
import numpy as np
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    print("MediaPipe not available, using fallback")
    mp = None
from collections import deque
import time
import json

class MediaPipeHSEmotionSystem:
    def __init__(self):
        """MediaPipe-based emotion recognition system"""
        # Check MediaPipe availability
        if mp is None:
            raise ImportError("MediaPipe not available. Install with: pip install mediapipe")
        
        # Initialize MediaPipe Face Detection using Tasks API
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.components import processors
            
            # Create face detector using new Tasks API
            base_options = python.BaseOptions(model_asset_path=r"F:\HSemotion\blaze_face_short_range.tflite")
            options = vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=0.5
            )
            self.face_detection = vision.FaceDetector.create_from_options(options)
            self.use_tasks_api = True
            print("[OK] MediaPipe face detection initialized (Tasks API)")
        except ImportError as e:
            raise RuntimeError(f"MediaPipe Tasks API not available: {e}. Try: pip uninstall mediapipe && pip install mediapipe")
        
        # Initialize HSEmotion model
        import torch
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
        
        from hsemotion.facial_emotions import HSEmotionRecognizer
        self.model = HSEmotionRecognizer(model_name='enet_b0_8_va_mtl')
        
        # Patch for 'conv_s2d' and 'aa' AttributeError due to timm version mismatch
        # The loaded model object lacks this attribute which new timm classes expect
        # We apply this to ALL modules safely
        if hasattr(self.model, 'model'):
            # Convert generator to list to avoid runtime modification issues during iteration
            for name, m in list(self.model.model.named_modules()):
                # Check for missing attributes in any module that might need them
                if not hasattr(m, 'conv_s2d'):
                    m.conv_s2d = None
                if not hasattr(m, 'aa'):
                    m.aa = torch.nn.Identity()
        
        torch.load = original_load
        
        # System variables
        self.va_history = deque(maxlen=5)
        self.log_data = []
        self.last_analysis_time = 0
        self.analysis_interval = 0.5
        
        print("[OK] MediaPipe HSEmotion system initialized")
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe Tasks API"""
        # Convert frame to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect faces
        detection_result = self.face_detection.detect(mp_image)
        
        faces = []
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                ih, iw = frame.shape[:2]
                
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                w = int(bbox.width)
                h = int(bbox.height)
                
                # Ensure coordinates are within frame bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                faces.append((x, y, w, h))
        
        return faces
    
    def classify_emotion_from_va(self, valence, arousal):
        """Classify emotion based on VA values"""
        if valence > 0.2:
            if arousal > 0.5:
                return "Happy", 0.95
            elif arousal > 0.0:
                return "Pleasant", 0.90
            else:
                return "Calm", 0.85
        elif valence > -0.2:
            if arousal > 0.6:
                return "Surprise", 0.90
            elif arousal > 0.0:
                return "Neutral", 0.95
            else:
                return "Relaxed", 0.85
        else:
            if arousal > 0.5:
                return "Anger", 0.90
            elif arousal > 0.0:
                return "Disgust", 0.85
            else:
                return "Sad", 0.90
    
    def predict_va_and_emotion(self, face_img):
        """Predict VA and emotion from face image"""
        try:
            _, va_scores = self.model.predict_emotions(face_img, logits=True)
            
            if isinstance(va_scores, np.ndarray) and va_scores.shape[0] >= 10:
                valence = float(va_scores[8])
                arousal = float(va_scores[9])
                
                emotion, confidence = self.classify_emotion_from_va(valence, arousal)
                return valence, arousal, emotion, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
        
        return 0.0, 0.0, "Unknown", 0.0
    
    def stabilize_va(self, valence, arousal):
        """Temporal stabilization"""
        if self.va_history:
            last_v, last_a = self.va_history[-1]
            alpha = 0.3
            valence = alpha * valence + (1 - alpha) * last_v
            arousal = alpha * arousal + (1 - alpha) * last_a
        
        self.va_history.append((valence, arousal))
        return valence, arousal
    
    def run_analysis(self, source=0):
        """Run MediaPipe-based emotion analysis"""
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        current_valence, current_arousal = 0.0, 0.0
        current_emotion, current_confidence = "Neutral", 0.0
        
        print("Starting MediaPipe emotion analysis. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            
            # MediaPipe face detection
            faces = self.detect_faces_mediapipe(frame)
            
            # Check if should analyze
            should_analyze = (current_time - self.last_analysis_time) >= self.analysis_interval
            
            for (x, y, w, h) in faces:
                if should_analyze:
                    # Extract and process face
                    face_crop = frame[y:y+h, x:x+w]
                    if face_crop.size > 0:
                        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_crop_rgb, (224, 224))
                        
                        # Predict emotion
                        current_valence, current_arousal, current_emotion, current_confidence = self.predict_va_and_emotion(face_resized)
                        current_valence, current_arousal = self.stabilize_va(current_valence, current_arousal)
                        
                        print(f"Analysis {len(self.log_data)+1}: V={current_valence:.3f}, A={current_arousal:.3f}, Emotion={current_emotion} ({current_confidence:.2f})")                        
                        # Log data
                        self.log_data.append({
                            'timestamp': current_time,
                            'valence': current_valence,
                            'arousal': current_arousal,
                            'emotion': current_emotion,
                            'confidence': current_confidence
                        })
                        
                        self.last_analysis_time = current_time
                
                # Draw bounding box and info
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"V:{current_valence:.2f} A:{current_arousal:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"{current_emotion} ({current_confidence:.2f})", 
                           (x, y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show analysis count
            cv2.putText(frame, f"MediaPipe Analyses: {len(self.log_data)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('MediaPipe HSEmotion Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save logs
        with open('mediapipe_emotion_log.json', 'w') as f:
            json.dump(self.log_data, f, indent=2)
        
        print(f"MediaPipe analysis complete. {len(self.log_data)} analyses performed.")

if __name__ == "__main__":
    system = MediaPipeHSEmotionSystem()
    system.run_analysis()