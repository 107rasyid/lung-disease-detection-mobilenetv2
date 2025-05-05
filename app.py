# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Konfigurasi model & label
IMAGE_SIZE = 256
LABELS = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']

# Load model (gunakan cache agar tidak di-load ulang setiap kali)
@st.cache_resource
def load_segmentation_model():
    model_path = 'model-mobilenetv2.h5'  # Pastikan nama file sesuai
    model = load_model(model_path)
    return model

model = load_segmentation_model()

# Prediksi fungsi
def predict(image, model):
    image = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))) / 255.0
    image = image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    probabilities = model.predict(image).reshape(-1)
    pred = LABELS[np.argmax(probabilities)]
    return pred, {x: float(y) for x, y in zip(LABELS, probabilities)}

# UI Streamlit
st.title("Klasifikasi Gambar X-Ray Paru")
st.write("Model ini mengklasifikasikan gambar ke dalam 4 kelas: NORMAL, TUBERCULOSIS, PNEUMONIA, COVID19.")

uploaded_file = st.file_uploader("Upload gambar X-Ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Diupload", use_column_width=True)

    with st.spinner("Melakukan prediksi..."):
        prediction, probs = predict(image, model)

    st.success(f"Prediksi: **{prediction}**")

    # Tampilkan probabilitas sebagai grafik batang
    st.subheader("Probabilitas:")
    st.bar_chart(probs)