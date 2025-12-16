import cv2
from facial_landmarks import FacialLandmarkExtractor

def main():
    cap = cv2.VideoCapture(0)
    extractor = FacialLandmarkExtractor()
    
    # Set sampling rate (extract features every 3rd frame for 30fps -> 10fps feature rate)
    extractor.set_sample_rate(3)
    
    print("Optimized Demo - Features extracted every 3rd frame")
    print("Press 'q' to quit, '1-5' to change sample rate, 's' to save features")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract landmarks (every frame for smooth visualization)
        landmarks = extractor.extract_landmarks(frame)
        
        if landmarks is not None:
            # Draw mesh (every frame)
            frame = extractor.draw_triangular_mesh(frame, landmarks)
            
            # Extract features (sampled rate)
            features = extractor.get_geometric_features(landmarks)
            if features:
                frame = extractor.draw_features_overlay(frame, features)
        
        cv2.imshow('Optimized Facial Features', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            rate = int(chr(key))
            extractor.set_sample_rate(rate)
            print(f"Sample rate set to: every {rate} frame(s)")
        elif key == ord('s'):
            # Save features
            extractor.save_features_to_file('features.csv', 'csv')
            extractor.save_features_to_file('features.npz', 'npy')
            print(f"Buffer contains {len(extractor.feature_buffer)} feature vectors")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()