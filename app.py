from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

app = Flask(__name__)

# Load the trained model
model = load_model('model/trained_model.h5')

# Load class names
with open('dataset_structure.json', 'r') as f:
    class_names = json.load(f)

# Preprocess uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    result = model.predict(img_array)
    predicted_class_index = np.argmax(result)

    # Map index to class name
    if str(predicted_class_index) in class_names:
        prediction = class_names[str(predicted_class_index)]
        response = {'class': prediction, 'confidence': float(result[0][predicted_class_index])}
    else:
        response = {'error': 'Predicted class index is out of range!'}

    # Clean up the uploaded file
    os.remove(file_path)

    return jsonify(response)

# Home route for testing
@app.route('/')
def home():
    return render_template('index.html')

# Added Flask backend changes for file processing
@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400

    # Save the file temporarily
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Process the file (reuse the prediction logic)
    img_array = preprocess_image(file_path)
    result = model.predict(img_array)
    predicted_class_index = np.argmax(result)

    # Map index to class name
    if str(predicted_class_index) in class_names:
        prediction = class_names[str(predicted_class_index)]
        response = {'message': f'Predicted: {prediction}, Confidence: {float(result[0][predicted_class_index]):.2f}'}
    else:
        response = {'message': 'Error: Predicted class index out of range!'}

    # Clean up uploaded file
    os.remove(file_path)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
