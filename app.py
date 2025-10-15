from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from fase1 import show_data
from model import build_CNN
from train import preprocess_and_train
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cargar datos y modelo una vez
x_train, y_train, x_test, y_test = show_data()
model = build_CNN()
preprocess_and_train(model, x_train, y_train)

# Helper para convertir imágenes numpy a archivos temporales
def save_images(images, folder):
    paths = []
    for idx, img in enumerate(images):
        img_path = os.path.join(folder, f"img_{idx}.png")
        Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
        paths.append(f"img_{idx}.png")
    return paths

@app.route('/')
def index():
    img_names = save_images(x_test[:10], app.config['UPLOAD_FOLDER'])
    return render_template('index.html', img_names=img_names)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    # Preprocesar imagen para el modelo
    img = Image.open(filepath).resize((x_train.shape[1], x_train.shape[2]))
    img_arr = np.array(img) / 255.0
    if img_arr.ndim == 2:  # grayscale
        img_arr = np.expand_dims(img_arr, axis=-1)
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = model.predict(img_arr)
    pred_class = np.argmax(pred, axis=1)[0]
    return render_template('predict.html', img_name=filename, pred_class=pred_class)