import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os

print("\n" + "="*70)
print("DEEPFAKE DETECTION - LIVE DEMO")
print("="*70)

model = load_model("saved_models/xception_proper_best.h5")
detector = MTCNN()

print("\nModel loaded: xception_proper_best.h5")
print("Accuracy: 89.25%")

def quick_test(video_path, label):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions = []
    
    indices = np.linspace(0, total - 1, 5, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = detector.detect_faces(rgb)
        
        if dets:
            d = max(dets, key=lambda x: x["box"][2]*x["box"][3])
            x, y, w, h = d["box"]
            m = int(0.2*min(w,h))
            face = frame[max(0,y-m):min(frame.shape[0],y+h+m), max(0,x-m):min(frame.shape[1],x+w+m)]
            if face.size > 0:
                face = cv2.resize(face, (224,224)) / 255.0
                pred = model.predict(np.expand_dims(face, axis=0), verbose=0)[0][0]
                predictions.append(pred)
    
    cap.release()
    
    if predictions:
        avg = np.mean(predictions)
        verdict = "REAL" if avg < 0.5 else "FAKE"
        conf = abs(avg - 0.5) * 2 * 100
        print(f"  {label:20} → {verdict:5} ({conf:.1f}% confidence)")

print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)

# Test a few videos
test_videos = [
    ("dataset/train/real/000.mp4", "Real video 1"),
    ("dataset/train/fake/000_003.mp4", "Fake video 1"),
    ("dataset/test/real/800.mp4", "Real video 2"),
    ("dataset/test/fake/801_870.mp4", "Fake video 2"),
]

for path, label in test_videos:
    if os.path.exists(path):
        quick_test(path, label)

print("="*70)
print("✅ Demo complete!")
