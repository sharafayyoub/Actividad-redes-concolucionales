# main.py
from fase1 import show_data
from model import build_CNN
from train import preprocess_and_train
from app import app

if __name__ == "__main__":
    app.run(debug=True)
