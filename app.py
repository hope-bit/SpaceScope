from flask import Flask, request, render_template, send_from_directory
import os
import numpy as np
from keras._tf_keras.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = load_model(r'C:\Users\HopeW\Downloads\SpaceScope\galaxy_mnist_classifierH5.h5')

# Define the class names
classArray = ["smooth_round", "smooth_cigar", "edge_on_disk", "barred_spiral"]  # Class names for labels

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image):
    image = image.resize((64, 64))  # Resize to the input size of your model
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Open and preprocess the image
        image = Image.open(file_path)
        image_preprocessed = preprocess_image(image)
        
        # Predict the class
        prediction = model.predict(image_preprocessed)
        class_id = np.argmax(prediction)
        class_name = classArray[class_id]  # Get the class name
        
        # Debugging information
        print(f"Prediction: {prediction}")
        print(f"Class ID: {class_id}")
        print(f"Class Name: {class_name}")
        
        return render_template('result.html', class_name=class_name, image_path=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)