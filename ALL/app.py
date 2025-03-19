import streamlit as st
from fastai.vision.all import *
from PIL import Image

# Load the exported model
learn = load_learner('AllModel.pkl')

# App title
st.title("Acute Lymphoblastic Luekemia Detection ")

# File uploader
uploaded_file = st.file_uploader("Upload an Image of WBC Malignant", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict the class
    img = PILImage.create(uploaded_file)
    pred, pred_idx, probs = learn.predict(img)
    st.write(f"### Prediction: {pred}")
    st.write(f"### Probability: {probs[pred_idx]:.4f}")
