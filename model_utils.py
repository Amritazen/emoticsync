# model_utils.py
import os
import zipfile
import numpy as np
import cv2
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam

EMOTIONS = [
    "Neutral", "Happy", "Sad", "Angry", "Surprise",
    "Fear", "Disgust", "Calm", "Energetic", "Excited"
]

def extract_zip_to(zip_path, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(target_dir)

def find_deepest_dir_with_ext(base_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".wav", ".mp3")):
    found = []
    for root, dirs, files in os.walk(base_dir):
        if any(f.lower().endswith(extensions) for f in files):
            found.append(root)
    if not found:
        return None
    best = max(found, key=lambda p: sum(1 for f in os.listdir(p) if f.lower().endswith(extensions)))
    return best

def load_face_data(dataset_dir, target_size=(48,48)):
    images, labels = [], []
    for root, dirs, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            path = os.path.join(root, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, target_size)
            label = os.path.basename(root)
            images.append(img.astype("float32")/255.0)
            labels.append(label)
    if len(images) == 0:
        return None, None, None
    X = np.array(images).reshape(-1, target_size[0], target_size[1], 1)
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    y = to_categorical(y_enc, num_classes=len(le.classes_))
    return X, y, le

def load_audio_data(dataset_dir, duration=3, sr=22050, n_mfcc=40):
    features, labels = [], []
    for root, dirs, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                continue
            path = os.path.join(root, fname)
            try:
                y, _ = librosa.load(path, sr=sr, duration=duration)
                if y.shape[0] == 0:
                    continue
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                mfcc_scaled = np.mean(mfcc.T, axis=0)
                features.append(mfcc_scaled)
                label = os.path.basename(root)
                labels.append(label)
            except Exception as e:
                print("Skipped audio:", path, "err:", e)
                continue
    if len(features) == 0:
        return None, None, None
    X = np.array(features)
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    y = to_categorical(y_enc, num_classes=len(le.classes_))
    return X, y, le

def build_face_model(input_shape=(48,48,1), n_classes=len(EMOTIONS)):
    inp = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation='relu')(inp)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_audio_model(input_dim=40, n_classes=len(EMOTIONS)):
    inp = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return loss, acc
