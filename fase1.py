import numpy as np
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt

# mantener referencias a las figuras para que no se cierren por GC
_DISPLAY_FIGS = []

def show_data():
    print("Cargando CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    print("Guardando algunas imágenes de ejemplo en 'uploads' (la web las mostrará) y mostrándolas en pantalla (no bloqueante)...")

    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    examples = []
    for i in range(3, 6):
        fname = f"example_{i}.png"
        fpath = os.path.join(upload_dir, fname)
        if not os.path.exists(fpath):
            img_arr = x_train[i]
            if img_arr.dtype != 'uint8':
                img_arr = (img_arr * 255).astype('uint8')
            Image.fromarray(img_arr).save(fpath)
        examples.append((i, os.path.join(upload_dir, fname)))

    # Mostrar las imágenes en ventanas matplotlib sin bloquear la ejecución
    try:
        for idx, path in examples:
            img = Image.open(path)
            arr = np.array(img)
            fig = plt.figure(figsize=(3,3))
            plt.imshow(arr)
            plt.title(f"Example {idx}")
            plt.axis('off')
            _DISPLAY_FIGS.append(fig)
        # show non-blocking: devuelve inmediatamente y mantiene las ventanas abiertas
        plt.show(block=False)
    except Exception as e:
        print("No se pudo mostrar imágenes en pantalla (no bloqueante):", e)

    return x_train, y_train, x_test, y_test
