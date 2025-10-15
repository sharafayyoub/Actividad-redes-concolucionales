# main.py
from fase1 import show_data
from model import build_CNN
from train import preprocess_and_train

if __name__ == "__main__":
    # Fase 1: Cargar y mostrar datos
    x_train, y_train, x_test, y_test = show_data()

    # Fase 2: Construir el modelo
    model = build_CNN()

    # Fase 3: Preprocesar y entrenar
    preprocess_and_train(model, x_train, y_train)
