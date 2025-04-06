from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load your saved model
model = tf.keras.models.load_model("mnist_model.h5")


@app.route('/')
def index():
    return "Backend is running!"

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    data = pd.read_csv(file)

    # Assumption: data is preprocessed like training (e.g., shape = [?, 784])
    predictions = model.predict(data)
    predicted_labels = np.argmax(predictions, axis=1)

    return jsonify({'predictions': predicted_labels.tolist()})
