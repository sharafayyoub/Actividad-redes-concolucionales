# app.py
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
from fase1 import show_data
from model import build_CNN
from train import preprocess_and_train
from werkzeug.utils import secure_filename
from PIL import Image

# Configuración de Flask
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------
# Cargar datos y entrenar modelo
# ---------------------------
x_train, y_train, x_test, y_test = show_data()
model = build_CNN()
preprocess_and_train(model, x_train, y_train, epochs=10)  # Entrenamiento con más epochs

# ---------------------------
# Helper: guardar imágenes de numpy array a archivos
# ---------------------------
def save_images(images, folder):
    paths = []
    for idx, img in enumerate(images):
        img_path = os.path.join(folder, f"img_{idx}.png")
        Image.fromarray((img * 255).astype(np.uint8)).save(img_path)
        paths.append(f"img_{idx}.png")
    return paths

# ---------------------------
# Rutas Flask
# ---------------------------

# Página principal: mostrar primeras 10 imágenes
@app.route('/')
def index():
    img_names = save_images(x_test[:10], app.config['UPLOAD_FOLDER'])
    return render_template('index.html', img_names=img_names)

# Servir imágenes de uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Ruta de predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    # Guardar archivo subido
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocesar imagen para el modelo
    img = Image.open(filepath).resize((x_train.shape[1], x_train.shape[2]))
    img_arr = np.array(img).astype('float32') / 255.0

    # Asegurar que tenga 3 canales (RGB)
    if img_arr.ndim == 2:  # si es grayscale
        img_arr = np.stack([img_arr]*3, axis=-1)
    elif img_arr.shape[2] == 4:  # si tiene alpha
        img_arr = img_arr[:, :, :3]

    # Añadir dimensión batch
    img_arr = np.expand_dims(img_arr, axis=0)

    # Predicción
    pred = model.predict(img_arr)
    pred_class = np.argmax(pred, axis=1)[0]

    # Renderizar resultado
    return render_template('predict.html', img_name=filename, pred_class=pred_class)
