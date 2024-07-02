import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load your trained model
model_path = 'pneumoniaDetectLV17.h5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((128, 128))  # Resize image to match model's expected sizing
    
    # Check if image has 3 channels (RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')  # Convert to RGB if not already
    
    img = np.asarray(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit app
def main():
    st.title('Pneumonia Detection App')
    st.write('Upload a chest X-ray image for pneumonia detection')

    # File upload
    uploaded_file = st.file_uploader("Choose a chest X-ray image ...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(uploaded_file)

        # Predict the class (0: Normal, 1: Pneumonia)
        prediction = model.predict(processed_image)
        pneumonia_probability = prediction[0][0]

        if pneumonia_probability > 0.5:
            st.write(f'Prediction: Pneumonia (Probability: {pneumonia_probability:.2f})')
        else:
            st.write(f'Prediction: Normal (Probability: {1 - pneumonia_probability:.2f})')

# Run the app
if __name__ == '__main__':
    main()
