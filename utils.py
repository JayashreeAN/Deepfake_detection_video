import cv2
import numpy as np
from mtcnn import MTCNN
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

class VideoProcessor:
    def __init__(self, face_size=(224, 224)):
        self.face_size = face_size
        self.detector = MTCNN()
    
    def extract_frames(self, video_path, max_frames=30, skip_frames=5):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        extracted = 0
        while cap.isOpened() and extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % skip_frames == 0:
                frames.append(frame)
                extracted += 1
            frame_count += 1
        cap.release()
        return frames
    
    def detect_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_frame)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        x, y, w, h = face['box']
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = w + 2 * margin
        h = h + 2 * margin
        face_crop = rgb_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, self.face_size)
        return face_resized
    
    def process_video(self, video_path, max_frames=30):
        frames = self.extract_frames(video_path, max_frames=max_frames)
        faces = []
        for frame in frames:
            face = self.detect_face(frame)
            if face is not None:
                face = face.astype('float32') / 255.0
                faces.append(face)
        return np.array(faces)

def prepare_dataset(data_dir, target_size=(224, 224)):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, zoom_range=0.2, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(os.path.join(data_dir, 'train'), target_size=target_size, batch_size=32, class_mode='categorical', shuffle=True)
    test_generator = test_datagen.flow_from_directory(os.path.join(data_dir, 'test'), target_size=target_size, batch_size=32, class_mode='categorical', shuffle=False)
    return train_generator, test_generator

def aggregate_predictions(predictions, method='average'):
    if method == 'average':
        avg_pred = np.mean(predictions, axis=0)
        final_class = np.argmax(avg_pred)
        confidence = avg_pred[final_class]
    else:
        frame_classes = np.argmax(predictions, axis=1)
        final_class = np.bincount(frame_classes).argmax()
        confidence = np.sum(frame_classes == final_class) / len(frame_classes)
    return final_class, confidence
