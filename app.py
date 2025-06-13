
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="MNIST Digit Classifier")

st.title("🧠 MNIST Digit Classifier")
st.write("Upload a 28x28 image of a handwritten digit.")

# Load the model
@st.cache_resource
def load_cnn_model():
    return load_model("mnist_cnn_model.h5")

model = load_cnn_model()

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = ImageOps.invert(image)  # Make background black and digit white
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    st.image(image, caption="Processed Image", width=150)
    st.success(f"### Predicted Digit: {np.argmax(prediction)}")
