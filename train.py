# fase3_train.py
from skimage import color
from tensorflow.keras.utils import to_categorical

def preprocess_and_train(model, x_train, y_train, epochs=5):
    print("Preprocesando datos (grayscale y normalización)...")
    grey_xtrain = color.rgb2gray(x_train)
    grey_xtrain = grey_xtrain / 255.0
    grey_xtrain = grey_xtrain[..., None]  # expand dims para el canal

    ytrain_cat = to_categorical(y_train)

    print("Compilando y entrenando el modelo...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(grey_xtrain, ytrain_cat, epochs=25, batch_size=128, validation_split=0.1)
    model.save("modelo_cifar10.h5")
print("Modelo guardado como modelo_cifar10.h5")
