import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# --------------------
# PIXELATE (PIL Version)
# --------------------
def pixelate(img, pixel_size=12):
    small = img.resize(
        (img.width // pixel_size, img.height // pixel_size),
        resample=Image.NEAREST
    )
    return small.resize(img.size, Image.NEAREST)


# --------------------
# PENCIL SKETCH (PIL Version)
# --------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inverted = ImageOps.invert(gray)
    blur = inverted.filter(ImageFilter.GaussianBlur(radius=15))

    # Dodge blend
    blend = ImageOps.blend(
        gray,
        blur,
        alpha=0.1
    )
    return blend


# --------------------
# BLUR BACKGROUND (Face stays sharp)
# --------------------
def blur_background(img):
    # convert to array for masking
    np_img = np.array(img)

    # simple center crop mask: keeps face area sharp
    h, w = np_img.shape[:2]

    # face rectangle (approx center)
    cx, cy = w // 2, h // 2
    fw, fh = int(w * 0.55), int(h * 0.55)

    mask = Image.new("L", (w, h), 0)
    face_box = Image.new("L", (fw, fh), 255)
    mask.paste(face_box, (cx - fw//2, cy - fh//2))

    # blur whole image
    blurred = img.filter(ImageFilter.GaussianBlur(radius=20))

    # composite sharp face on blurred background
    return Image.composite(img, blurred, mask)


# --------------------
# DISPLAY OUTPUT
# --------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate(img), caption="Pixelated Face", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
