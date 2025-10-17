import numpy as np
import tensorflow as tf
import os
from PIL import Image

def _grayscale(img):
    arr = img.astype('float32')
    return arr[..., 0]*0.2989 + arr[..., 1]*0.5870 + arr[..., 2]*0.1140

def _sharpness_score(gray):
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return np.var(grad_mag)

def show_data():
    print("Cargando CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    print("Seleccionando imágenes más nítidas y guardándolas en 'uploads' para la interfaz...")

    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # Configuración: cuántas imágenes elegir y tamaño de muestra para evaluar rapidez
    top_k = 6            # cuántas imágenes nítidas guardar
    sample_size = 2000   # cuántas imágenes muestrear del conjunto de entrenamiento (ajustable)

    total = len(x_train)
    rng = np.random.RandomState(0)
    if sample_size < total:
        indices = rng.choice(total, size=sample_size, replace=False)
    else:
        indices = np.arange(total)

    scores = []
    for idx in indices:
        img = x_train[idx]
        gray = _grayscale(img)
        score = _sharpness_score(gray)
        scores.append((score, idx))

    # Ordenar por nitidez descendente y tomar top_k índices
    scores.sort(reverse=True, key=lambda x: x[0])
    top_indices = [idx for (_, idx) in scores[:top_k]]

    # Guardar las imágenes más nítidas como example_0.png ... example_{k-1}.png
    for rank, idx in enumerate(top_indices):
        fname = f"example_{rank}.png"
        fpath = os.path.join(upload_dir, fname)
        img_arr = x_train[idx]
        # asegurar uint8 en [0..255]
        if np.issubdtype(img_arr.dtype, np.floating):
            img_arr = (img_arr * 255.0).clip(0,255).astype('uint8')
        else:
            img_arr = img_arr.astype('uint8')
        Image.fromarray(img_arr).save(fpath)

    print(f"Guardadas {len(top_indices)} imágenes nítidas en '{upload_dir}'.")

    return x_train, y_train, x_test, y_test
