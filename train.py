# train.py
import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_and_train(model, x_train, y_train, epochs=7):
    print("Preprocesando datos (normalización y codificación)...")

    # Normalización (manteniendo 3 canales)
    x_train = x_train.astype('float32') / 255.0

    # Codificación one-hot (CIFAR-10 tiene 10 clases)
    y_train = to_categorical(y_train, 10)

    print("Compilando y entrenando el modelo...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.1)

