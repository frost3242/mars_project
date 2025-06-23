import streamlit as st
import librosa
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os
import h5py
import json
from tensorflow.keras.models import model_from_json

@st.cache_resource
def load_ser_model():
    # --- Load and patch model config ---
    with h5py.File("trained4_model.h5", "r") as f:
        model_config = f.attrs.get("model_config").decode("utf-8")
    config_dict = json.loads(model_config)

    # Patch InputLayer to use 'batch_input_shape' instead of 'batch_shape'
    for layer in config_dict["config"]["layers"]:
        if layer["class_name"] == "InputLayer":
            if "batch_shape" in layer["config"]:
                layer["config"]["batch_input_shape"] = layer["config"].pop("batch_shape")

    # Load model from fixed config
    model = model_from_json(json.dumps(config_dict))
    model.load_weights("trained4_model.h5")

    # Load class names
    classes = np.load("classes.npy")
    return model, classes
    
# ==== Constants ====
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 216

# ==== Load scaler (fit on training data) ====
@st.cache_resource
def load_scaler():
    return StandardScaler()

scaler = load_scaler()

# ==== Feature Extraction Function ====
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
    zcr = librosa.feature.zero_crossing_rate(y).T
    rms = librosa.feature.rms(y=y).T

    def pad(x):
        return x[:MAX_LEN] if x.shape[0] >= MAX_LEN else np.pad(x, ((0, MAX_LEN - x.shape[0]), (0, 0)), mode='constant')

    mfcc = pad(mfcc)
    chroma = pad(chroma)
    zcr = pad(zcr)
    rms = pad(rms)

    return np.concatenate([mfcc, chroma, zcr, rms], axis=1)

# ==== Streamlit UI ====
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a WAV file and get the predicted emotion.")

uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])

if uploaded_file is not None:
    file_path = "temp.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file, format='audio/wav')

    try:
        features = extract_features(file_path)
        features = features.reshape(1, -1)
        features = scaler.transform(features)  # for demo only; use training scaler in prod
        features = features.reshape(1, MAX_LEN, -1)
        prediction = model.predict(features)
        predicted_class = classes[np.argmax(prediction)]

        st.success(f"🔊 Predicted Emotion: **{predicted_class}**")
    except Exception as e:
        st.error(f"Error processing file: {e}")

    os.remove(file_path)
