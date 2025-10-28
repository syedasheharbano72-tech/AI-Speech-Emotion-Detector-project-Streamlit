import streamlit as st
import librosa
import numpy as np
import joblib
import soundfile as sf
import tempfile

# -----------------------------
# Load Pre-trained Model (You can replace with your own trained model)
# -----------------------------
# For demo: We'll simulate a trained model using simple logic
# You can replace this section with: model = joblib.load("emotion_model.pkl")

class DummyEmotionModel:
    def predict(self, features):
        emotions = ["Happy", "Sad", "Angry", "Calm", "Fearful"]
        return [np.random.choice(emotions)]

model = DummyEmotionModel()

# -----------------------------
# Extract audio features function
# -----------------------------
def extract_features(y, sr):
    # Compute basic features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    
    # Combine all features into a single vector
    features = np.hstack([mfcc, chroma, mel, contrast])
    return features

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎧 AI Speech Emotion Detector")
st.markdown("Upload an audio file (e.g., `.wav`) and let the AI detect the speaker's emotion.")

uploaded_file = st.file_uploader("Upload your audio file:", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save uploaded audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_path = temp_audio.name

    # Load audio and extract features
    try:
        y, sr = librosa.load(temp_path, sr=None)
        features = extract_features(y, sr).reshape(1, -1)
        prediction = model.predict(features)[0]

        st.subheader("🎯 Predicted Emotion:")
        st.success(prediction)
    except Exception as e:
        st.error(f"Error processing audio: {e}")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Librosa + AI Model")
