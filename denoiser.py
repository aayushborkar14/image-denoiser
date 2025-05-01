import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
from dncnn import denoise_image, DnCNN

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigma = 25  # Standard deviation of Gaussian noise

torch.classes.__path__ = []


# Load model
@st.cache_resource
def load_model():
    model = DnCNN()
    model.load_state_dict(torch.load("models/dncnn_epoch_100.pth", map_location=device))
    model.eval()
    model.to(device)
    return model


model = load_model()

# --- App title ---
st.title("Image Denoiser")
st.write(
    "Upload a grayscale image to see the denoising effect of the trained DnCNN model."
)

# --- File upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image using PIL and convert to grayscale numpy array
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)

    st.subheader("Original Image")
    st.image(img, use_container_width=True, caption="Grayscale input image")

    # Add Gaussian noise
    noise = np.random.randn(*img.shape) * sigma
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype("uint8")

    st.subheader("Noisy Image")
    st.image(noisy_img, use_container_width=True, caption="Image with Gaussian noise")

    # Denoise using DnCNN
    denoised_img = denoise_image(noisy_img, model)

    st.subheader("Denoised Image")
    st.image(
        denoised_img, use_container_width=True, caption="Image after denoising by DnCNN"
    )

    # Side-by-side comparison
    comparison = np.hstack((img, noisy_img, denoised_img))
    st.subheader("Side-by-Side Comparison")
    st.image(
        comparison, caption="Original | Noisy | Denoised", use_container_width=True
    )
