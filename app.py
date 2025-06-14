from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load the trained model
model = load_model("mnist_cnn_model.h5")

@app.get("/")
def read_root():
    return {"message": "MNIST FastAPI is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    predictions = model.predict(img_array)
    digit = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    return JSONResponse(content={
        "digit": digit,
        "confidence": confidence
    })
