import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def set_seed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def check_gpu():
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))

def load_and_preprocess_data():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=10):
    print("\nTraining the model...\n")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_data=(x_test, y_test)
    )
    return history

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("\nTest Accuracy:", test_acc)

def save_model(model, path="mnist_model.h5"):
    model.save(path)
    print(f"Model saved to {path}")

def plot_predictions(model, x_sample):
    prediction = model.predict(np.expand_dims(x_sample, axis=0))
    plt.imshow(x_sample, cmap='gray')
    plt.title(f"Predicted Label: {np.argmax(prediction)}")
    plt.show()

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.show()

def main():
    set_seed()
    check_gpu()

    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = build_model()
    model.summary()

    history = train_model(model, x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)
    save_model(model)
    plot_predictions(model, x_test[0])
    plot_training_history(history)

if __name__ == "__main__":
    main()
