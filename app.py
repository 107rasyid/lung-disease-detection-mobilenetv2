# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Konfigurasi model & label
IMAGE_SIZE = 256
LABELS = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']

# Load model (gunakan cache agar tidak diload ulang setiap kali)
@st.cache_resource
def load_segmentation_model():
    model_path = 'model-mobilenetv2.h5'  # Pastikan file ini ada di direktori yang sama
    model = load_model(model_path)
    return model

model = load_segmentation_model()

# Fungsi prediksi
def predict(image, model):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    probabilities = model.predict(image).reshape(-1)
    pred = LABELS[np.argmax(probabilities)]
    return pred, {label: float(prob) for label, prob in zip(LABELS, probabilities)}

# UI Streamlit
st.title("Klasifikasi Gambar X-Ray Paru")
st.markdown("""
Model ini akan mengklasifikasikan gambar X-Ray menjadi salah satu dari 4 kategori:

- **NORMAL**
- **TUBERCULOSIS**
- **PNEUMONIA**
- **COVID19**
""")

uploaded_file = st.file_uploader("Upload gambar X-Ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Diupload", use_column_width=True)

    with st.spinner("Melakukan prediksi..."):
        prediction, probs = predict(image, model)

    st.success(f"Prediksi: **{prediction}**")

    st.subheader("Probabilitas Kelas:")
    st.bar_chart(probs)