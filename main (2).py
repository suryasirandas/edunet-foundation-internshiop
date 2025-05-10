from flask import Flask, request, jsonify
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

# Load your trained model
model = load_model('cnn_tess_data.h5')  # Make sure to specify the correct path

# Initialize Flask app
app = Flask(__name__)

# Endpoint to process the uploaded audio file
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Load the audio file
        audio, sr = librosa.load(file, duration=3)  # Adjust the duration if needed
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        
        # Reshape and prepare the input for the model
        mfcc_mean = mfcc_mean.reshape((1, 13, 1, 1))  # Change this depending on your model's input shape
        
        # Get the prediction from the model
        prediction = model.predict(mfcc_mean)
        emotion = np.argmax(prediction)  # Get the emotion label based on the highest probability
        
        # Emotion labels (Modify based on your model)
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
        
        return jsonify({"emotion": emotion_labels[emotion]})

if __name__ == '__main__':
    app.run(debug=True)
