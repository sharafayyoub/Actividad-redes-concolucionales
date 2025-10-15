import numpy as np
import tensorflow as tf
from skimage import color
import matplotlib.pyplot as plt

def show_data():
    print("Cargando CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    print("Mostrando algunas imágenes de ejemplo...")
    def plot_images(color_img, grayscale_img):
        plt.figure(figsize=(8,8))
        plt.subplot(1,2,1)
        plt.title('Color Image', color='green', fontsize=15)
        plt.imshow(color_img)
        plt.subplot(1,2,2)
        plt.title('Grayscale Image', color='black', fontsize=15)
        plt.imshow(grayscale_img, cmap='gray')
        plt.show()

    for i in range(3,6):
        plot_images(x_train[i], color.rgb2gray(x_train[i]))

    return x_train, y_train, x_test, y_test
