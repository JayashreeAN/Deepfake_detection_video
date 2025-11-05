import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os

model = load_model("saved_models/xception_proper_best.h5")
detector = MTCNN()

def test_any_video(video_path):
    if not os.path.exists(video_path):
        print(f"\nERROR: Video not found!")
        return
    
    print("\n" + "="*70)
    print("DEEPFAKE DETECTION TEST")
    print("="*70)
    print(f"Video: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("ERROR: Cannot open!")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Resolution: {width}x{height}")
    print(f"Frames: {total_frames} ({total_frames/fps:.1f}s)")
    
    predictions = []
    faces_detected = 0
    
    print("Extracting faces...")
    
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_frame)
        
        if detections:
            detection = max(detections, key=lambda x: x["box"][2] * x["box"][3])
            x, y, w, h = detection["box"]
            margin = int(0.2 * min(w, h))
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face = frame[y1:y2, x1:x2]
            
            if face.size > 0:
                face = cv2.resize(face, (224, 224))
                face = face / 255.0
                pred = model.predict(np.expand_dims(face, axis=0), verbose=0)[0][0]
                predictions.append(pred)
                faces_detected += 1
    
    cap.release()
    
    print(f"Faces detected: {faces_detected}")
    
    if len(predictions) > 0:
        avg_pred = np.mean(predictions)
        
        print("\n" + "="*70)
        print("RESULT")
        print("="*70)
        
        # Better thresholds
        if avg_pred < 0.35:
            verdict = "REAL (High Confidence)"
            confidence = (1 - avg_pred) * 100
        elif avg_pred > 0.65:
            verdict = "FAKE - DEEPFAKE (High Confidence)"
            confidence = avg_pred * 100
        elif avg_pred < 0.5:
            verdict = "REAL (Low Confidence)"
            confidence = (1 - avg_pred) * 100
        else:
            verdict = "FAKE (Low Confidence)"
            confidence = avg_pred * 100
        
        print(f"Verdict: {verdict}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Raw Score: {avg_pred:.4f}")
        print("="*70)
        
        print("\nSCORE INTERPRETATION:")
        print(f"  0.0 - 0.35 = DEFINITELY REAL")
        print(f"  0.35 - 0.50 = PROBABLY REAL")
        print(f"  0.50 - 0.65 = PROBABLY FAKE")
        print(f"  0.65 - 1.0 = DEFINITELY FAKE")
        
    else:
        print("\nERROR: No faces detected!")

print("="*70)
print("DEEPFAKE DETECTION SYSTEM v2")
print("="*70)

video_path = input("\nEnter video path: ").strip()

if video_path.startswith(chr(34)):
    video_path = video_path[1:-1]

test_any_video(video_path)
