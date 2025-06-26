import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image, ImageOps
import os

# Title
st.title("Handwritten Digit Recognition with MNIST")
st.write("Draw a digit (0-9) or upload one, and the app will predict it!")

MODEL_FILE = "mnist_model.h5"

@st.cache_resource
def train_model():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.fit(x_train, y_train, epochs=3, validation_split=0.1, verbose=0)
    model.save(MODEL_FILE)
    return model

# Load or train model
if not os.path.exists(MODEL_FILE):
    st.info("Training model, please wait...")
model = train_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image of a digit (28x28, grayscale):", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=False, width=150)

    if st.button("Predict"):
        img_array = preprocess_image(image)
        prediction = model.predict(img_array)
        st.success(f"Predicted Digit: {np.argmax(prediction)}")
