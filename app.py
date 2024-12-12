from flask import Flask, request, render_template
import h5py
import numpy as np
from keras._tf_keras.keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = load_model(r'C:\Users\HopeW\Downloads\SpaceScope\galaxy_mnist_classifierH5.h5')

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
        image = Image.open(file)
        image = preprocess_image(image)
        prediction = model.predict(image)
        class_id = np.argmax(prediction)
        return f'Predicted class: {class_id}'

if __name__ == '__main__':
    app.run(debug=True)