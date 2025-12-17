import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model, Model
from PIL import Image

app = Flask(__name__)
model = None

# Constants (Must match training!)
IMG_SIZE = (75, 75) 

def load_our_model():
    global model
    try:
        model = load_model('final_model.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0) # (1, 75, 75, 3)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train first.'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Predict
        prediction = model.predict(processed_image)
        prob = float(prediction[0][0])
        
        # Threshold 0.5
        label = "Malignant (Cancer Detected)" if prob > 0.5 else "Normal (Healthy)"
        confidence = prob if prob > 0.5 else 1 - prob
        
        return jsonify({
            'label': label,
            'confidence': f"{confidence*100:.2f}%",
            'probability': prob
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_our_model()
    # Debug=True helps seeing errors, but reload might re-load model
    app.run(debug=True, port=5000)
