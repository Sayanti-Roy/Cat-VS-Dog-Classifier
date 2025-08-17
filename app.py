import streamlit as st 
import tensorflow as tf 
import numpy as np 
from keras.preprocessing import image
from PIL import Image

# Load model
model = tf.keras.models.load_model('cat_dog_classifier.h5')

# Streamlit UI
st.title("Cat vs Dog Classifier")
st.write("Upload an image, and the model will predict whether it's a cat or a dog.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image...", use_container_width=True)
    
    # Preprocessing
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    prediction = model.predict(img_array)
    result = "Dog" if prediction[0][0] > 0.5 else "Cat"
    
    st.subheader("Prediction:")
    st.success(f"The model predicts: **{result}**")