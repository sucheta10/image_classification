import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import pathlib
from pyngrok import ngrok


# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('my_model2.hdf5')
    return model

# @st.cache-decorator is used to cache the results of the 'load_model' function.
# 'allow_output_mutation=True' allows the cached model object to be mutated

with st.spinner('Model is being loaded..'):
    model = load_model()
# While the model is being loaded (load_model function), a spinner (st.spinner) is shown in the Streamlit app interface.
st.write("""
         # Sports Event Classification
         """) # Displays a title "Sports Event Classification" using st.write

file = st.file_uploader("Please upload a sports image file", type=["jpg", "png"]) # creates a file uploader widget where users can upload an image file
st.set_option('deprecation.showfileUploaderEncoding', False) # ensures correct handling of file upload encoding


# Define a function to predict the class of the image
def import_and_predict(image_data, model):
    size = (180, 180) # Resizes the image
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image) # Converts the image to a NumPy array
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # adjusts its color channels
    img_reshape = img[np.newaxis, ...] # Reshapes the image to match the model's input shape
    prediction = model.predict(img_reshape) # Uses the loaded model to predict the class of the image and returns the prediction
    return prediction


# Load class names (assuming the dataset is in a local directory)
data_dir = pathlib.Path("C:/Users/Sucheta/OneDrive/Desktop/python/100 sports image classification/train")  #  is used to specify the directory path where class directories are stored
class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()]) #  It then retrieves the class names by listing subdirectories

if file is None:
    st.text("Please upload a sports image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True) # Opens and displays the image
    predictions = import_and_predict(image, model) #  to predict the class of the uploaded image
    score = tf.nn.softmax(predictions[0])# Converts the model's prediction scores to probabilities
    # Displays the predicted class name and confidence level
    st.write("Prediction:", class_names[np.argmax(score)])
    st.write("Confidence: {:.2f}%".format(100 * np.max(score)))

# Connect Ngrok
public_url = ngrok.connect(8501)  # Port number for running the app
st.write('Public URL:', public_url)