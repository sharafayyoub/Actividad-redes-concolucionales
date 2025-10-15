# fase2_model.py
from keras import layers
from keras.models import Sequential

def build_CNN():
    print("Construyendo el modelo CNN...")
    model = Sequential([
        layers.Input(shape=(32,32,1)),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPool2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    model.summary()
    return model
