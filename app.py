from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "MNIST Digit Classifier API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' in request.files:
            # Case 1: image file uploaded
            file = request.files['file']
            img = Image.open(file).convert('L')
        else:
            # Case 2: base64 image in JSON
            data = request.get_json(force=True)
            img_data = base64.b64decode(data['image'])
            img = Image.open(io.BytesIO(img_data)).convert('L')

        # Resize and preprocess image
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))

        return jsonify({'prediction': predicted_digit})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
