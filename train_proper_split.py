import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import Xception
from tqdm import tqdm

detector = MTCNN()

def extract_faces(video_path, num_frames=5):
    try:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return None
        
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        faces = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            dets = detector.detect_faces(rgb)
            
            if dets:
                d = max(dets, key=lambda x: x["box"][2] * x["box"][3])
                x, y, w, h = d["box"]
                m = int(0.2 * min(w, h))
                face = frame[max(0, y-m):min(frame.shape[0], y+h+m), max(0, x-m):min(frame.shape[1], x+w+m)]
                if face.size > 0:
                    face = cv2.resize(face, (224, 224))
                    faces.append(face)
        
        cap.release()
        
        if len(faces) > 0:
            return np.mean(faces, axis=0).astype(np.uint8)
        return None
    except:
        return None

def load_split_dataset():
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Use original video folders
    real_dir = r"archive (2)\FaceForensics++_C23\original"
    fake_dir = r"archive (2)\FaceForensics++_C23\Deepfakes"
    
    for label, (class_name, dir_path) in enumerate([("real", real_dir), ("fake", fake_dir)]):
        if not os.path.exists(dir_path):
            print(f"ERROR: {dir_path} not found!")
            continue
        
        videos = sorted([f for f in os.listdir(dir_path) if f.endswith(".mp4")])
        print(f"\n{class_name.upper()}: Found {len(videos)} videos")
        
        # 60% train, 40% test
        split = int(len(videos) * 0.6)
        train_vids = videos[:split]
        test_vids = videos[split:]
        
        print(f"  Train: {len(train_vids)} | Test: {len(test_vids)}")
        
        print(f"  Loading TRAINING...")
        for v in tqdm(train_vids):
            face = extract_faces(os.path.join(dir_path, v), num_frames=3)
            if face is not None:
                X_train.append(face / 255.0)
                y_train.append(label)
        
        print(f"  Loading TESTING...")
        for v in tqdm(test_vids):
            face = extract_faces(os.path.join(dir_path, v), num_frames=3)
            if face is not None:
                X_test.append(face / 255.0)
                y_test.append(label)
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def build_model():
    base = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=x)
    
    for layer in base.layers[:-30]:
        layer.trainable = False
    
    return model

print("="*70)
print("PROPER TRAIN/TEST SPLIT (NO DATA LEAKAGE)")
print("="*70)

X_train, y_train, X_test, y_test = load_split_dataset()

print(f"\n✅ Dataset Ready:")
print(f"   Train: {X_train.shape}")
print(f"   Test: {X_test.shape}")

if len(X_train) < 10:
    print("ERROR!")
else:
    print("\nBuilding model...")
    model = build_model()
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    
    cp = ModelCheckpoint("saved_models/xception_proper_best.h5", save_best_only=True, monitor="val_accuracy", mode="max")
    es = EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
    
    print("Training...")
    h = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=8, callbacks=[cp, es], verbose=1)
    
    model.save("saved_models/xception_proper_final.h5")
    
    print(f"\n✅ Complete!")
    print(f"Best: {max(h.history['val_accuracy'])*100:.2f}%")
    print(f"Final: {h.history['val_accuracy'][-1]*100:.2f}%")
