import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os

model = load_model("saved_models/xception_quick_best.h5")
detector = MTCNN()

def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Cannot open: {video_path}")
        return
    
    predictions = []
    print(f"\nTesting: {os.path.basename(video_path)}")
    
    for idx in np.linspace(0, total_frames - 1, 5, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_frame)
        
        if detections:
            d = max(detections, key=lambda x: x["box"][2] * x["box"][3])
            x, y, w, h = d["box"]
            margin = int(0.2 * min(w, h))
            face = frame[max(0, y - margin):min(frame.shape[0], y + h + margin), max(0, x - margin):min(frame.shape[1], x + w + margin)]
            if face.size > 0:
                face = cv2.resize(face, (224, 224)) / 255.0
                pred = model.predict(np.expand_dims(face, axis=0), verbose=0)[0][0]
                predictions.append(pred)
    
    cap.release()
    
    if predictions:
        avg = np.mean(predictions)
        verdict = "FAKE" if avg > 0.5 else "REAL"
        conf = abs(avg - 0.5) * 2 * 100
        print(f"Result: {verdict} (Confidence: {conf:.2f}%)")

real_videos = [f for f in os.listdir("dataset/train/real") if f.endswith(".mp4")][:2]
fake_videos = [f for f in os.listdir("dataset/train/fake") if f.endswith(".mp4")][:2]

print("\nTesting REAL videos:")
for v in real_videos:
    detect_deepfake(os.path.join("dataset/train/real", v))

print("\n\nTesting FAKE videos:")
for v in fake_videos:
    detect_deepfake(os.path.join("dataset/train/fake", v))
