import os
import numpy as np
import librosa
import joblib
import tensorflow as tf

#  CONFIG 
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 216
MODEL_PATH = "trained4_model.h5"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
CLASSES_PATH = "classes.npy"

#  Load Assets 
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
classes = np.load(CLASSES_PATH)

#  Feature Extraction 
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
    zcr = librosa.feature.zero_crossing_rate(y).T
    rms_feature = librosa.feature.rms(y=y).T

    def pad(x):
        return x[:MAX_LEN] if x.shape[0] >= MAX_LEN else np.pad(x, ((0, MAX_LEN - x.shape[0]), (0, 0)), mode='constant')

    mfcc = pad(mfcc)
    chroma = pad(chroma)
    zcr = pad(zcr)
    rms_feature = pad(rms_feature)

    return np.concatenate([mfcc, chroma, zcr, rms_feature], axis=1)

#  Prediction Function 
def predict_emotion(file_path):
    try:
        features = extract_features(file_path)
        features = features.reshape(1, *features.shape)
        flat = features.reshape(1, -1)
        scaled = scaler.transform(flat).reshape(features.shape)
        prediction = model.predict(scaled)
        pred_label = np.argmax(prediction, axis=1)
        emotion = label_encoder.inverse_transform(pred_label)[0]
        print(f"[✔] File: {os.path.basename(file_path)} --> Predicted Emotion: {emotion}")
        return emotion
    except Exception as e:
        print(f"[✘] Error processing {file_path}: {e}")
        return None

# Batch Test Folder 
if __name__ == "__main__":
    TEST_FOLDER = "test_audio"  # i assumed this folder where test .wav files are placed
    if not os.path.exists(TEST_FOLDER):
        print(f"[✘] Test folder '{TEST_FOLDER}' not found!")
        exit(1)

    for file in os.listdir(TEST_FOLDER):
        if file.endswith(".wav"):
            file_path = os.path.join(TEST_FOLDER, file)
            predict_emotion(file_path)
