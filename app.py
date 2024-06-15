import streamlit as st
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess_input
from keras.preprocessing import image as keras_image
from PIL import Image
import matplotlib.pyplot as plt
import json


# Load the models
cnn_model = load_model('fashion_mnist_model_cnn.h5')
vgg16_model = load_model('fashion_mnist_model2_87_56563vgg16.h5')


with open("cnn_model_history.json", 'r') as f:
    cnn_history = json.load(f)

with open("vgg16_model_history.json", 'r') as f1:
    vgg_history = json.load(f1)


# Function to process image for CNN Model
def process_image_cnn(image):
    img = Image.open(image)
    img = img.resize((28, 28))
    img = img.convert('L')  # grayscale
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array


# Function to process image for VGG16 Model
def process_image_vgg(image):
    img = Image.open(image)
    img = img.resize((56, 56))
    img = img.convert('RGB')  # ensure image is RGB
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = vgg_preprocess_input(img_array)
    return img_array

# Function to plot training history
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['loss'], label='Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Loss over Epochs')

    ax2.plot(history['accuracy'], label='Accuracy')
    ax2.plot(history['val_accuracy'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Accuracy over Epochs')

    st.pyplot(fig)


# Function to plot probabilities
def plot_probabilities(probabilities, class_names):
    fig, ax = plt.subplots()
    ax.barh(class_names, probabilities)
    ax.set_xlabel('Probability (%)')
    ax.set_title('Class Probabilities')
    st.pyplot(fig)


# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', ['Prediction', 'Visualizations', 'About'])

if options == 'Prediction':  # Prediction page
    st.markdown("# :rainbow[Fashion MNIST Image Classification]")

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    # Model selection
    model_option = st.selectbox('Which model would you like to use?', ('CNN Model', 'VGG16 Model'))

    # User inputs: image
    image = st.file_uploader('Upload an image:', type=['jpg', 'jpeg', 'png'])
    if image is not None:
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Classify Image'):
            with st.spinner('Model working....'):
                if model_option == 'CNN Model':
                    img_array = process_image_cnn(image)
                    prediction = cnn_model.predict(img_array).argmax()
                    prediction1 = cnn_model.predict(img_array)
                else:
                    img_array = process_image_vgg(image)
                    prediction = vgg16_model.predict(img_array).argmax()
                    prediction1 = vgg16_model.predict(img_array)
                st.success(f'Prediction: {class_names[prediction]}')
                st.subheader('Class Probabilities:')
                probabilities = [float(prediction1[0][i]) * 100 for i in range(len(class_names))]
                st.json({class_names[i]: probabilities[i] for i in range(len(class_names))})
                plot_probabilities(probabilities, class_names)

elif options == 'Visualizations':  # Visualizations page
    st.markdown("# Visualizations")

    # Initialize session state for history
    if 'cnn_history' not in st.session_state:
        st.session_state['cnn_history'] = cnn_history
    if 'vgg_history' not in st.session_state:
        st.session_state['vgg_history'] = vgg_history

    model_history_option = st.selectbox('Which model history would you like to see?',
                                        ('CNN Model History', 'VGG16 Model History'))

    if model_history_option == 'CNN Model History':
        plot_history(st.session_state['cnn_history'])
    else:
        plot_history(st.session_state['vgg_history'])


elif options == 'About':  # About page
    st.markdown("# About")
    st.write("""
        This application uses two models to classify images from the Fashion MNIST dataset:
        1. A custom Convolutional Neural Network (CNN).
        2. A pre-trained VGG16 model fine-tuned on the Fashion MNIST dataset.

        The dataset contains images of various types of clothing and accessories. The models predict the category of the item in the uploaded image.
    """)
