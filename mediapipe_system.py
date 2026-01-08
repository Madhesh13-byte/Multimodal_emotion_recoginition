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
        if hasattr(self.model, 'model'):
            for name, m in list(self.model.model.named_modules()):
                if not hasattr(m, 'conv_s2d'):
                    m.conv_s2d = None
                if not hasattr(m, 'aa'):
                    m.aa = torch.nn.Identity()
        
        torch.load = original_load
        
        # System variables
        self.emotion_history = {}  # Track emotion history per face
        self.last_emotion = {}  # Track last logged emotion per face
        self.emotion_buffer = {}  # Buffer for consistent emotion detection
        self.log_data = []
        self.last_analysis_time = 0
        self.analysis_interval = 0.3  # Faster analysis for better tracking
        
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
    
    def should_log_emotion(self, face_id, emotion, confidence):
        """Check if emotion change is significant enough to log with consistency check"""
        # Skip neutral emotions (noise reduction)
        if emotion == "Neutral":
            return False
        
        # Initialize buffer for new face
        if face_id not in self.emotion_buffer:
            self.emotion_buffer[face_id] = deque(maxlen=5)
        
        # Add current emotion to buffer
        self.emotion_buffer[face_id].append(emotion)
        
        # Check if we have 5 consistent emotions
        if len(self.emotion_buffer[face_id]) >= 5:
            # Check if all 5 recent emotions are the same
            recent_emotions = list(self.emotion_buffer[face_id])
            if all(e == recent_emotions[0] for e in recent_emotions):
                consistent_emotion = recent_emotions[0]
                
                # Log if it's different from last logged emotion
                if face_id not in self.last_emotion or self.last_emotion[face_id] != consistent_emotion:
                    if confidence > 0.3:  # Confidence threshold
                        self.last_emotion[face_id] = consistent_emotion
                        return True
        
        return False
    
    def smooth_emotion(self, face_id, emotion, confidence):
        """Apply temporal smoothing to emotion predictions"""
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=3)
        
        self.emotion_history[face_id].append((emotion, confidence))
        
        # Use majority voting for stability
        if len(self.emotion_history[face_id]) >= 2:
            emotions = [e[0] for e in self.emotion_history[face_id]]
            confidences = [e[1] for e in self.emotion_history[face_id]]
            
            # Return most frequent emotion with average confidence
            from collections import Counter
            most_common = Counter(emotions).most_common(1)[0][0]
            avg_confidence = np.mean([c for e, c in self.emotion_history[face_id] if e == most_common])
            return most_common, avg_confidence
        
        return emotion, confidence
    
    def predict_direct_emotion(self, face_img):
        """Predict emotion using HSEmotion model"""
        try:
            # Use the working method from before
            _, va_scores = self.model.predict_emotions(face_img, logits=True)
            
            if isinstance(va_scores, np.ndarray) and va_scores.shape[0] >= 10:
                # Get 8-emotion scores (indices 0-7)
                emotion_scores = va_scores[:8]
                emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                emotion_idx = np.argmax(emotion_scores)
                emotion = emotions[emotion_idx]
                confidence = float(emotion_scores[emotion_idx])
                
                return emotion, confidence
        except Exception as e:
            print(f"Prediction error: {e}")
        
        return "Unknown", 0.0
    
    def run_analysis(self, source=0):
        """Run MediaPipe-based emotion analysis"""
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting MediaPipe emotion analysis. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            faces = self.detect_faces_mediapipe(frame)
            should_analyze = (current_time - self.last_analysis_time) >= self.analysis_interval
            
            for i, (x, y, w, h) in enumerate(faces):
                face_emotion, face_confidence = "Neutral", 0.0
                
                if should_analyze:
                    face_crop = frame[y:y+h, x:x+w]
                    if face_crop.size > 0:
                        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_crop_rgb, (224, 224))
                        
                        # Predict emotion for this specific face
                        raw_emotion, raw_confidence = self.predict_direct_emotion(face_resized)
                        
                        # Apply temporal smoothing
                        face_emotion, face_confidence = self.smooth_emotion(i, raw_emotion, raw_confidence)
                        
                        # Always show current emotion in terminal
                        print(f"Face {i+1}: Emotion={face_emotion} ({face_confidence:.2f})")
                        
                        # Only log significant emotion changes (not neutral noise)
                        if self.should_log_emotion(i, face_emotion, face_confidence):
                            print(f"Face {i+1}: Emotion={face_emotion} ({face_confidence:.2f}) [LOGGED]")
                            
                            # Log data for this face
                            self.log_data.append({
                                'timestamp': current_time,
                                'face_id': i+1,
                                'emotion': face_emotion,
                                'confidence': face_confidence
                            })
                else:
                    # Use last known emotion for display
                    if i in self.emotion_history and len(self.emotion_history[i]) > 0:
                        face_emotion, face_confidence = self.emotion_history[i][-1]
                
                # Draw individual face info
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{face_emotion} ({face_confidence:.2f})", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if should_analyze and faces:
                self.last_analysis_time = current_time
            
            # Show analysis count
            cv2.putText(frame, f"MediaPipe Analysis: {len(faces)} faces", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('MediaPipe HSEmotion Analysis', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save logs
        with open('mediapipe_emotion_log.json', 'w') as f:
            json.dump(self.log_data, f, indent=2)
        
        print(f"MediaPipe analysis complete. {len(self.log_data)} emotion changes logged.")

if __name__ == "__main__":
    system = MediaPipeHSEmotionSystem()
    system.run_analysis()