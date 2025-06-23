import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import os
import json
import h5py
import joblib
from tensorflow.keras.models import model_from_json

# ==== Load Trained Model ====
@st.cache_resource
def load_ser_model():
    try:
        with h5py.File("trained4_model.h5", "r") as f:
            raw_config = f.attrs.get("model_config")
            model_config = raw_config.decode("utf-8") if isinstance(raw_config, bytes) else raw_config
        config_dict = json.loads(model_config)

        for layer in config_dict["config"]["layers"]:
            if layer["class_name"] == "InputLayer" and "batch_shape" in layer["config"]:
                layer["config"]["batch_input_shape"] = layer["config"].pop("batch_shape")

        model = model_from_json(json.dumps(config_dict))
        model.load_weights("trained4_model.h5")

        classes = np.load("classes.npy", allow_pickle=True)
        return model, classes
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None, None

model, classes = load_ser_model()
if model is None or classes is None:
    st.stop()

# ==== Load Pre-Fitted Scaler ====
@st.cache_resource
def load_scaler():
    try:
        scaler = joblib.load("scaler.pkl")
        return scaler
    except Exception:
        st.warning("⚠️ Could not load scaler.pkl, using default StandardScaler (not recommended)")
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()

scaler = load_scaler()

# ==== Constants ====
SAMPLE_RATE = 22050
N_MFCC = 40
MAX_LEN = 216

# ==== Feature Extraction ====
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
    zcr = librosa.feature.zero_crossing_rate(y).T
    rms = librosa.feature.rms(y=y).T

    def pad(x):
        return x[:MAX_LEN] if x.shape[0] >= MAX_LEN else np.pad(x, ((0, MAX_LEN - x.shape[0]), (0, 0)), mode='constant')

    return np.concatenate([pad(mfcc), pad(chroma), pad(zcr), pad(rms)], axis=1)

# ==== Streamlit UI ====
st.title("🎤 Speech Emotion Recognition")
st.write("Upload a WAV file to predict the emotion.")

uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])

if uploaded_file is not None:
    file_path = "temp.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(uploaded_file, format='audio/wav')

    try:
        features = extract_features(file_path)
        features = features.reshape(1, -1)
        features = scaler.transform(features)  # Use pre-fitted scaler
        features = features.reshape(1, MAX_LEN, -1)

        prediction = model.predict(features)
        predicted_class = classes[np.argmax(prediction)]

        st.success(f"🔊 Predicted Emotion: **{predicted_class}**")
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
    finally:
        os.remove(file_path)
