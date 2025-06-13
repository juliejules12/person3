import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("mnist_cnn_model.h5")

st.title("MNIST Digit Recognizer")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('L').resize((28,28))
    img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img_array)
    st.write(f"Prediction: **{np.argmax(prediction)}**")
