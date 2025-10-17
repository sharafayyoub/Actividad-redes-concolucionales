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
preprocess_and_train(model, x_train, y_train, epochs=7)  # 🔹 Entrenamiento con 7 epochs

# ---------------------------
# Helper: guardar imágenes de numpy array a archivos (robusto)
# ---------------------------
def save_images(images, folder):
    paths = []
    for idx, img in enumerate(images):
        img_name = f"img_{idx}.png"
        img_path = os.path.join(folder, img_name)
        try:
            if isinstance(img, np.ndarray):
                arr = img
                # float -> [0,255] uint8
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr * 255.0).clip(0,255).astype('uint8')
                else:
                    arr = arr.astype('uint8')
                # grayscale -> stack to 3 channels
                if arr.ndim == 2:
                    arr = np.stack([arr]*3, axis=-1)
                # RGBA -> drop alpha
                if arr.shape[-1] == 4:
                    arr = arr[:, :, :3]
                Image.fromarray(arr).save(img_path)
            else:
                # si ya es PIL Image u otro objeto con save()
                img.save(img_path)
        except Exception as e:
            print(f"Warning: no se pudo guardar imagen {idx}: {e}")
            placeholder = (np.zeros((32,32,3), dtype='uint8'))
            Image.fromarray(placeholder).save(img_path)
        paths.append(os.path.basename(img_path))
    return paths

# ---------------------------
# Diccionario de clases CIFAR-10
# ---------------------------
CIFAR10_CLASSES = {
    0: "avión",
    1: "automóvil",
    2: "pájaro",
    3: "gato",
    4: "ciervo",
    5: "perro",
    6: "rana",
    7: "caballo",
    8: "barco",
    9: "camión"
}

# ---------------------------
# Rutas Flask
# ---------------------------

# Página principal: mostrar primeras 10 imágenes
@app.route('/')
def index():
    upload_dir = app.config['UPLOAD_FOLDER']
    # Buscar imágenes de ejemplo guardadas por fase1.py
    example_imgs = sorted([f for f in os.listdir(upload_dir)
                           if f.startswith('example_') and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if example_imgs:
        img_names = example_imgs
    else:
        # respaldo: guardar x_test[:10] si no hay imágenes de ejemplo
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

    # Preprocesar imagen para el modelo: forzar RGB y 32x32, normalizar
    try:
        img = Image.open(filepath).convert('RGB').resize((32, 32))
        img_arr = np.array(img).astype('float32') / 255.0
        # Asegurar forma (32,32,3)
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr]*3, axis=-1)
        if img_arr.shape[-1] == 4:
            img_arr = img_arr[:, :, :3]
        img_arr = np.expand_dims(img_arr, axis=0)  # (1,32,32,3)
    except Exception as e:
        print(f"Error al preprocesar la imagen subida: {e}")
        return redirect(url_for('index'))

    # Predicción: obtener top-1 y top-3 con confianza
    pred = model.predict(img_arr)
    probs = pred[0]
    # top-3
    top_k = 3
    top_idxs = probs.argsort()[::-1][:top_k]
    top_preds = []
    for idx in top_idxs:
        name = CIFAR10_CLASSES.get(int(idx), str(int(idx)))
        conf = float(probs[int(idx)]) * 100.0
        top_preds.append((name, round(conf, 2)))

    # principal
    pred_name, pred_conf = top_preds[0]

    # Renderizar resultado con nombre de clase, confianza y top-3
    return render_template('predict.html',
                           img_name=filename,
                           pred_class=pred_name,
                           confidence=pred_conf,
                           top_preds=top_preds)
