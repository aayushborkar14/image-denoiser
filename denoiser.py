import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
import os
from dncnn import denoise_image as denoise_image_dncnn, DnCNN
from dncnn_rl import denoise_image as denoise_image_dncnn_rl, DnCNNRL
from rednet import denoise_image as denoise_image_rednet, REDNet30
from rednet_bam import (
    denoise_image as denoise_image_rednet_bam,
    REDNet30_CBAM as REDNet30Bam,
)
from rednet_bam2 import (
    denoise_image as denoise_image_rednet_bam2,
    REDNet30_CBAM as REDNet30Bam2,
)
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigma = 25  # Standard deviation of Gaussian noise

# Fix torch.classes error if it appears during Streamlit hot-reload
torch.classes.__path__ = []


# --- Load models ---
@st.cache_resource
def load_dncnn():
    model = DnCNN()
    model.load_state_dict(torch.load("models/dncnn_epoch_100.pth", map_location=device))
    model.eval()
    model.to(device)
    return model


@st.cache_resource
def load_dncnn_rl():
    model = DnCNNRL()
    model.load_state_dict(
        torch.load("models/dncnn_epoch_100_rl.pth", map_location=device)
    )
    model.eval()
    model.to(device)
    return model


@st.cache_resource
def load_rednet():
    model = REDNet30()
    model.load_state_dict(torch.load("models/rednet_epoch_20.pth", map_location=device))
    model.eval()
    model.to(device)
    return model


@st.cache_resource
def load_rednet_bam():
    model = REDNet30Bam()
    model.load_state_dict(
        torch.load("models/rednet_cbam_epoch_70.pth", map_location=device)
    )
    model.eval()
    model.to(device)
    return model


@st.cache_resource
def load_rednet_bam2():
    model = REDNet30Bam2()
    model.load_state_dict(
        torch.load("models/rednet_cbam2_epoch_100.pth", map_location=device)
    )
    model.eval()
    model.to(device)
    return model


def crop_image_to_multiple_of(img, multiple):
    h, w = img.shape[:2]
    new_h = h - (h % multiple)
    new_w = w - (w % multiple)
    return img[:new_h, :new_w]


# --- App title ---
st.title("Image Denoiser")
st.write(
    "Choose a sample or upload a grayscale image to see the denoising effect of a trained model."
)

# --- Model selection ---
model_choice = st.selectbox(
    "Choose a model:",
    ["DnCNN", "DnCNN RL", "REDNet30", "REDNet30 CBAM", "REDNet30 CBAM2"],
)
if model_choice == "DnCNN":
    model = load_dncnn()
    denoise_func = denoise_image_dncnn
elif model_choice == "DnCNN RL":
    model = load_dncnn_rl()
    denoise_func = denoise_image_dncnn_rl
elif model_choice == "REDNet30":
    model = load_rednet()
    denoise_func = denoise_image_rednet
elif model_choice == "REDNet30 CBAM":
    model = load_rednet_bam()
    denoise_func = denoise_image_rednet_bam
elif model_choice == "REDNet30 CBAM2":
    model = load_rednet_bam2()
    denoise_func = denoise_image_rednet_bam2

# --- Image selection ---
st.subheader("Choose Image Source")
image_source = st.radio("Select image input method:", ("Sample Image", "Upload Image"))

if image_source == "Sample Image":
    sample_dir = "sample_images"
    image_names = sorted(
        [
            f
            for f in os.listdir(sample_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    if not image_names:
        st.warning("No sample images found in the directory.")
        st.stop()
    selected_image_name = st.selectbox("Choose a sample image:", image_names)
    image_path = os.path.join(sample_dir, selected_image_name)
    image = Image.open(image_path)
elif image_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload your image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        st.warning("Please upload an image to proceed.")
        st.stop()

# --- Image Preprocessing ---
if "REDNet30" not in model_choice:
    image = image.convert("L")
img = np.array(image)
if "REDNet30" in model_choice:
    img = crop_image_to_multiple_of(img, 32)

st.subheader("Original Image")
st.image(img, use_container_width=True, caption="Grayscale input image")

# --- Add Gaussian Noise ---
noise = np.random.randn(*img.shape) * sigma
noisy_img = img + noise
noisy_img = np.clip(noisy_img, 0, 255).astype("uint8")

st.subheader("Noisy Image")
st.image(noisy_img, use_container_width=True, caption="Image with Gaussian noise")

# --- Denoise ---
denoised_img = denoise_func(noisy_img, model)

st.subheader("Denoised Image")
st.image(
    denoised_img,
    use_container_width=True,
    caption=f"Image after denoising by {model_choice}",
)

# --- Side-by-side comparison ---
comparison = np.hstack((img, noisy_img, denoised_img))
st.subheader("Side-by-Side Comparison")
st.image(comparison, caption="Original | Noisy | Denoised", use_container_width=True)

# --- Metrics ---
st.subheader(
    f"PSNR: {peak_signal_noise_ratio(img, denoised_img, data_range=255.0):.2f}"
)
st.subheader(
    f"SSIM: {structural_similarity(img, denoised_img, data_range=255.0, channel_axis=-1):.4f}"
)

