# -*- coding: utf-8 -*-
import os
import gdown
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ====== Konfigurasi Model dari Google Drive ======
MODEL_PATH = 'resnet50_malnutrition.h5'
DRIVE_FILE_ID = '1LXqbHVrg2ZqFkdmf354f-i8pnXNP6UZ_'  # dari link kamu
DRIVE_URL = f'https://drive.google.com/uc?id={DRIVE_FILE_ID}'

if not os.path.exists(MODEL_PATH):
    print("Mengunduh model dari Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# ====== Setup Aplikasi Flask ======
os.makedirs('static/uploads', exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model dan label kelas
model = load_model(MODEL_PATH)
classes = ['Undernutrition', 'normal', 'overnutrition']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = Image.open(filepath).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    prediction = classes[np.argmax(preds)]

    image_url = filepath.replace('\\', '/')

    return render_template('result.html',
                           prediction=prediction,
                           image_path=image_url,
                           probs=preds,
                           classes=classes)

if __name__ == '__main__':
    app.run(debug=True)
