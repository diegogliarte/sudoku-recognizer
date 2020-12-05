# imports for array-handling and plotting
import numpy as np

# keras imports for the dataset and building our neural network
from keras.models import Sequential, load_model
import tensorflow as tf


mnist_model = load_model("digit_model.h5")

def predict_image(image):
    predicted_classes = mnist_model.predict(image)
    return np.argmax(predicted_classes, axis=1)