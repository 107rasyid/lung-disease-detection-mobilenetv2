import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.utils import custom_object_scope
import tempfile
import os

# --- Custom Layer & Metrics ---
class Cast(Layer):
    def __init__(self, **kwargs):
        super(Cast, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.cast(inputs, tf.keras.backend.floatx())
    def get_config(self):
        return super(Cast, self).get_config()

def precision_metric(y_true, y_pred):
    y_pred = K.cast(y_pred > 0.5, K.floatx())
    tp = K.sum(y_true * y_pred)
    pp = K.sum(y_pred)
    return (tp + 1e-10) / (pp + 1e-10)

def recall_metric(y_true, y_pred):
    y_pred = K.cast(y_pred > 0.5, K.floatx())
    tp = K.sum(y_true * y_pred)
    pos = K.sum(y_true)
    return (tp + 1e-10) / (pos + 1e-10)

def f1_metric(y_true, y_pred):
    p = precision_metric(y_true, y_pred)
    r = recall_metric(y_true, y_pred)
    return 2 * (p * r) / (p + r + 1e-10)

# --- Load Model ---
MODEL_PATH = "model/mobilenetv2.h5"
with custom_object_scope({
    'precision_metric': precision_metric,
    'recall_metric': recall_metric,
    'f1_metric': f1_metric,
    'Cast': Cast
}):
    model = load_model(MODEL_PATH)

# --- Utils ---
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def preprocess_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = apply_clahe(img)
    img = img.astype(np.float32) / 255.0
    return img

def classify_image(image, model):
    image_3ch = np.repeat(image[..., np.newaxis], 3, axis=-1)
    image_input = np.expand_dims(image_3ch, axis=0)
    predictions = model.predict(image_input)
    predicted_class = np.argmax(predictions)
    class_labels = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']
    return predicted_class, class_labels[predicted_class], predictions[0], class_labels

def generate_log(predictions, class_labels, predicted_label):
    log_text = f"Predicted Class: {predicted_label}\nClass Probabilities:\n"
    for i, prob in enumerate(predictions):
        log_text += f"{class_labels[i]}: {prob:.4f}\n"
    return log_text

# --- Streamlit UI ---
st.set_page_config(page_title="Klasifikasi Citra X-ray", layout="centered")
st.title("🩺 Aplikasi Klasifikasi Citra X-ray")
st.write("Upload citra X-ray (format .jpg, .png, .jpeg), dan sistem akan mengklasifikasikannya.")

uploaded_file = st.file_uploader("Upload File", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = preprocess_image(uploaded_file.read())

    st.image(image, caption="Citra setelah preprocessing", use_column_width=True, clamp=True, channels="GRAY")

    predicted_idx, predicted_label, predictions, class_labels = classify_image(image, model)

    st.markdown(f"### 🔍 Prediksi: `{predicted_label}`")
    
    st.subheader("📊 Probabilitas:")
    for i, prob in enumerate(predictions):
        st.write(f"- **{class_labels[i]}**: {prob:.4f}")

    # Tampilkan dan download log
    log_content = generate_log(predictions, class_labels, predicted_label)
    st.download_button(
        label="📥 Download Hasil Prediksi (.txt)",
        data=log_content,
        file_name="hasil_klasifikasi.txt",
        mime="text/plain"
    )