# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="mnist_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("MNIST Digit Recognition (.tflite)")
st.write("Upload a 28x28 grayscale image of a digit")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # TFLite needs float32 input with shape (1, 28, 28, 1)
    input_data = np.expand_dims(img_array, -1)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(prediction)

    st.image(image, caption='Processed Image', width=100)
    st.write(f"Predicted Digit: **{predicted_label}**")
