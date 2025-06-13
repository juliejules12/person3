import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import requests

MODEL_PATH = "mnist_cnn_model.h5"
MODEL_URL = "https://YOUR_PUBLIC_LINK_TO_MODEL"  # Replace this

# Download model if not present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

model = load_model(MODEL_PATH)

# Rest of your Streamlit app below...
