import cv2
import mediapipe as mp
import numpy as np
import time

# Feature vector ordering for ML consistency
FEATURE_ORDER = [
    'avg_ear', 'mar', 'norm_nose_mouth', 'norm_left_brow_eye',
    'norm_right_brow_eye', 'norm_mouth_width', 'ear_diff', 'brow_diff',
    'left_mouth_angle', 'right_mouth_angle', 'mouth_angle_diff',
    'delta_ear', 'delta_mar', 'delta_brow_diff', 'delta2_ear', 'delta2_mar'
]


try:
    from mediapipe.python.solutions.face_mesh_connections import (
        FACEMESH_TESSELATION,
        FACEMESH_LEFT_EYE,
        FACEMESH_RIGHT_EYE,
        FACEMESH_LEFT_EYEBROW,
        FACEMESH_RIGHT_EYEBROW,
        FACEMESH_LIPS
    )
except ImportError:
    from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION
    FACEMESH_LEFT_EYE = FACEMESH_TESSELATION
    FACEMESH_RIGHT_EYE = FACEMESH_TESSELATION
    FACEMESH_LEFT_EYEBROW = FACEMESH_TESSELATION
    FACEMESH_RIGHT_EYEBROW = FACEMESH_TESSELATION
    FACEMESH_LIPS = FACEMESH_TESSELATION


class FacialLandmarkExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # For temporal derivatives
        self.prev_features = None
        self.prev_time = None
        # Frame sampling control
        self.frame_count = 0
        self.sample_rate = 3  # Extract features every N frames
        self.last_features = None
        # Feature storage
        self.feature_buffer = []
        self.max_buffer_size = 1000  # Store last 1000 feature vectors
        
    def extract_landmarks(self, image):
        """Extract 468 facial landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # Convert to numpy array (468 landmarks x 3 coordinates)
            landmark_points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
            return landmark_points
        return None
    
    def get_key_points(self, landmarks):
        """Extract key facial regions for emotion recognition"""
        if landmarks is None:
            return None
            
        # Key landmark indices for emotion recognition
        key_indices = {
            'left_eye': [33, 246, 161, 160, 159, 158, 157, 173, 130, 131, 132, 133, 144, 145, 153, 154, 155],
            'right_eye': [362, 398, 384, 385, 386, 387, 388, 466, 263, 464, 368, 372, 373, 374, 380, 381, 382],
            'eyebrows': [46, 53, 52, 65, 55, 70, 276, 283, 282, 295, 285, 300],
            'nose': [1, 2, 3, 4, 5, 6, 19, 20, 94, 125, 141, 235, 236, 8, 9, 10, 151, 168, 197, 196],
            'mouth': [0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 78, 17, 18, 200, 199, 175]
        }
        
        key_landmarks = {}
        for region, indices in key_indices.items():
            key_landmarks[region] = landmarks[indices]
            
        return key_landmarks
    
    def draw_region_legend(self, image):
        """Draw color legend for facial regions"""
        legend_colors = [
            ('Eye', (255, 0, 0)),      # Blue
            # ('Right Eye', (255, 0, 0)),     # Blue
            # ('Left Eyebrow', (0, 255, 255)), # Yellow
            ('Eyebrow', (0, 255, 255)), # Yellow
            # ('Nose', (0, 255, 0)),         # Green
            ('Lips', (0, 0, 255)),         # Red
            ('Face', (255, 255, 0))      # Gray
        ]
        
        y_offset = 60
        for i, (region, color) in enumerate(legend_colors):
            y = y_offset + i * 25
            cv2.rectangle(image, (10, y-10), (30, y+5), color, -1)
            cv2.putText(image, region, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def draw_triangular_mesh(self, image, landmarks):
        """Draw color-coded triangular mesh using MediaPipe's predefined regions"""
        if landmarks is None:
            return image
            
        h, w = image.shape[:2]
        
        # Convert landmarks to pixel coordinates
        points = []
        for landmark in landmarks:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            points.append((x, y))
        
        # Define colors for different regions (BGR format)
        colors = {
            'left_eye': (255, 0, 0),       # Blue
            'right_eye': (255, 100, 0),    # Dark Blue
            'left_eyebrow': (0, 255, 255), # Yellow
            'right_eyebrow': (0, 200, 255), # Orange
            'lips': (0, 0, 255),           # Red
            'face': (255, 255, 0)        # Gray
        }
        
        # Track drawn connections to prevent duplicates
        drawn_connections = set()
        
        # Draw region-specific connections with priority order
        try:
            # 1. Left Eye (highest priority)
            for connection in FACEMESH_LEFT_EYE:
                i, j = connection
                if i < len(points) and j < len(points):
                    key = tuple(sorted([i, j]))
                    if key not in drawn_connections:
                        cv2.line(image, points[i], points[j], colors['left_eye'], 1)
                        drawn_connections.add(key)
            
            # 2. Right Eye
            for connection in FACEMESH_RIGHT_EYE:
                i, j = connection
                if i < len(points) and j < len(points):
                    key = tuple(sorted([i, j]))
                    if key not in drawn_connections:
                        cv2.line(image, points[i], points[j], colors['right_eye'], 1)
                        drawn_connections.add(key)
            
            # 3. Left Eyebrow
            for connection in FACEMESH_LEFT_EYEBROW:
                i, j = connection
                if i < len(points) and j < len(points):
                    key = tuple(sorted([i, j]))
                    if key not in drawn_connections:
                        cv2.line(image, points[i], points[j], colors['left_eyebrow'], 1)
                        drawn_connections.add(key)
            
            # 4. Right Eyebrow
            for connection in FACEMESH_RIGHT_EYEBROW:
                i, j = connection
                if i < len(points) and j < len(points):
                    key = tuple(sorted([i, j]))
                    if key not in drawn_connections:
                        cv2.line(image, points[i], points[j], colors['right_eyebrow'], 1)
                        drawn_connections.add(key)
            
            # 5. Lips
            for connection in FACEMESH_LIPS:
                i, j = connection
                if i < len(points) and j < len(points):
                    key = tuple(sorted([i, j]))
                    if key not in drawn_connections:
                        cv2.line(image, points[i], points[j], colors['lips'], 1)
                        drawn_connections.add(key)
        
        except:
            # Fallback if MediaPipe connections fail
            pass
        
        # 6. Draw remaining face mesh connections in gray
        for connection in FACEMESH_TESSELATION:
            i, j = connection
            if i < len(points) and j < len(points):
                key = tuple(sorted([i, j]))
                if key not in drawn_connections:
                    cv2.line(image, points[i], points[j], colors['face'], 1)
        
        return image
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two 3D points"""
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3 with numerical stability"""
        v1 = p1 - p2
        v2 = p3 - p2
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            return 0.0
            
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    def calculate_ear(self, left_corner, right_corner, top1, top2, bottom1, bottom2):
        """Calculate Eye Aspect Ratio using 6 key points with division safety"""
        # Vertical distances
        v1 = self.calculate_distance(top1, bottom1)
        v2 = self.calculate_distance(top2, bottom2)
        # Horizontal distance with safety
        h = max(self.calculate_distance(left_corner, right_corner), 1e-6)
        return (v1 + v2) / (2.0 * h)
    
    def calculate_mar(self, left_corner, right_corner, top, bottom):
        """Calculate Mouth Aspect Ratio using 4 key points with division safety"""
        # Vertical distance
        v = self.calculate_distance(top, bottom)
        # Horizontal distance with safety
        h = max(self.calculate_distance(left_corner, right_corner), 1e-6)
        return v / h
    
    def get_geometric_features(self, landmarks, include_temporal=True, force_extract=False):
        """Extract enhanced geometric features with adaptive sampling"""
        self.frame_count += 1
        
        # Skip frames for efficiency (except forced extraction)
        if not force_extract and self.frame_count % self.sample_rate != 0:
            return self.last_features
            
        return self._extract_features(landmarks, include_temporal)
    
    def _extract_features(self, landmarks, include_temporal=True):
        """Internal feature extraction logic"""
        if landmarks is None:
            return None
            
        current_time = time.time()
        features = {}
        
        # Basic measurements
        left_ear = self.calculate_ear(
            landmarks[33], landmarks[133], landmarks[159], landmarks[158], landmarks[145], landmarks[153]
        )
        right_ear = self.calculate_ear(
            landmarks[362], landmarks[263], landmarks[386], landmarks[385], landmarks[374], landmarks[380]
        )
        mar = self.calculate_mar(landmarks[61], landmarks[291], landmarks[13], landmarks[14])
        
        # Inter-eye distance for normalization with safety
        eye_distance = max(self.calculate_distance(landmarks[33], landmarks[362]), 1e-6)
        
        # Raw distances
        left_brow_eye = self.calculate_distance(landmarks[70], landmarks[159])
        right_brow_eye = self.calculate_distance(landmarks[300], landmarks[386])
        nose_mouth = self.calculate_distance(landmarks[2], landmarks[13])
        mouth_width = self.calculate_distance(landmarks[61], landmarks[291])
        
        # 1. NORMALIZED FEATURES (HIGH PRIORITY)
        features['avg_ear'] = (left_ear + right_ear) / 2
        features['mar'] = mar
        features['norm_nose_mouth'] = nose_mouth / eye_distance
        features['norm_left_brow_eye'] = left_brow_eye / eye_distance
        features['norm_right_brow_eye'] = right_brow_eye / eye_distance
        features['norm_mouth_width'] = mouth_width / eye_distance
        
        # 2. SYMMETRY FEATURES (Very important)
        features['ear_diff'] = abs(left_ear - right_ear)
        features['brow_diff'] = abs(left_brow_eye - right_brow_eye) / eye_distance
        
        # Mouth corner angles
        left_angle = self.calculate_angle(landmarks[2], landmarks[61], landmarks[13])
        right_angle = self.calculate_angle(landmarks[2], landmarks[291], landmarks[14])
        features['left_mouth_angle'] = left_angle
        features['right_mouth_angle'] = right_angle
        features['mouth_angle_diff'] = abs(left_angle - right_angle)
        
        # 3. TEMPORAL DERIVATIVES (FPS-aware)
        if include_temporal and self.prev_features is not None and self.prev_time is not None:
            dt = max(current_time - self.prev_time, 1e-3)  # FPS safety
            
            # First-order derivatives (velocity)
            delta_ear = (features['avg_ear'] - self.prev_features['avg_ear']) / dt
            delta_mar = (features['mar'] - self.prev_features['mar']) / dt
            delta_brow_diff = (features['brow_diff'] - self.prev_features['brow_diff']) / dt
            
            features['delta_ear'] = delta_ear
            features['delta_mar'] = delta_mar
            features['delta_brow_diff'] = delta_brow_diff
            
            # Second-order derivatives (acceleration)
            features['delta2_ear'] = delta_ear - self.prev_features.get('delta_ear', 0.0)
            features['delta2_mar'] = delta_mar - self.prev_features.get('delta_mar', 0.0)
        else:
            features['delta_ear'] = 0.0
            features['delta_mar'] = 0.0
            features['delta_brow_diff'] = 0.0
            features['delta2_ear'] = 0.0
            features['delta2_mar'] = 0.0
        
        # Store for next frame
        self.prev_features = features.copy()
        self.prev_time = current_time
        self.last_features = features.copy()
        
        # Store in buffer with timestamp
        self._store_features(features, current_time)
        
        return features
    
    def _store_features(self, features, timestamp):
        """Store features in buffer with timestamp"""
        feature_entry = {
            'timestamp': timestamp,
            'features': features.copy(),
            'vector': np.array([features[k] for k in FEATURE_ORDER])
        }
        
        self.feature_buffer.append(feature_entry)
        
        # Maintain buffer size
        if len(self.feature_buffer) > self.max_buffer_size:
            self.feature_buffer.pop(0)
    
    def get_feature_history(self, seconds=None, count=None):
        """Get stored features by time or count"""
        if not self.feature_buffer:
            return []
            
        if seconds is not None:
            cutoff_time = time.time() - seconds
            return [entry for entry in self.feature_buffer if entry['timestamp'] >= cutoff_time]
        elif count is not None:
            return self.feature_buffer[-count:]
        else:
            return self.feature_buffer.copy()
    
    def save_features_to_file(self, filename, format='csv'):
        """Save stored features to file"""
        if not self.feature_buffer:
            print("No features to save")
            return
            
        if format == 'csv':
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                header = ['timestamp'] + FEATURE_ORDER
                writer.writerow(header)
                # Data
                for entry in self.feature_buffer:
                    row = [entry['timestamp']] + entry['vector'].tolist()
                    writer.writerow(row)
        elif format == 'npy':
            # Save as numpy arrays
            timestamps = np.array([entry['timestamp'] for entry in self.feature_buffer])
            vectors = np.array([entry['vector'] for entry in self.feature_buffer])
            np.savez(filename, timestamps=timestamps, features=vectors, feature_names=FEATURE_ORDER)
            
        print(f"Saved {len(self.feature_buffer)} feature vectors to {filename}")
    
    def clear_buffer(self):
        """Clear stored features"""
        self.feature_buffer.clear()
    
    def set_sample_rate(self, rate):
        """Set feature extraction sample rate (1=every frame, 3=every 3rd frame)"""
        self.sample_rate = max(1, rate)
    
    def get_feature_vector(self, landmarks):
        """Get ML-ready feature vector with explicit ordering"""
        features = self.get_geometric_features(landmarks)
        if features is None:
            return None
            
        # Use explicit feature ordering for ML consistency
        return np.array([features[k] for k in FEATURE_ORDER])
    
    def draw_features_overlay(self, image, features):
        """Draw enhanced geometric features overlay on image"""
        if features is None:
            return image
            
        # Background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (350, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Enhanced feature text
        y_pos = 25
        cv2.putText(image, "NORMALIZED FEATURES", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(image, f"EAR: {features['avg_ear']:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        y_pos += 15
        cv2.putText(image, f"MAR: {features['mar']:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        y_pos += 15
        cv2.putText(image, f"Nose-Mouth: {features['norm_nose_mouth']:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        y_pos += 15
        cv2.putText(image, f"Mouth Width: {features['norm_mouth_width']:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        y_pos += 25
        cv2.putText(image, "SYMMETRY FEATURES", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(image, f"EAR Diff: {features['ear_diff']:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        y_pos += 15
        cv2.putText(image, f"Brow Diff: {features['brow_diff']:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        y_pos += 15
        cv2.putText(image, f"Angle Diff: {features['mouth_angle_diff']:.3f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        y_pos += 25
        cv2.putText(image, "TEMPORAL FEATURES", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 20
        cv2.putText(image, f"ΔEAR: {features['delta_ear']:.4f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        y_pos += 15
        cv2.putText(image, f"ΔMAR: {features['delta_mar']:.4f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        y_pos += 15
        cv2.putText(image, f"Δ²EAR: {features['delta2_ear']:.4f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
        y_pos += 15
        cv2.putText(image, f"Δ²MAR: {features['delta2_mar']:.4f}", (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
        
        return image