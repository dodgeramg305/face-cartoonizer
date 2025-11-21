import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

# --------------------------------------------------
# 1. Pixelate face (Pillow only)
# --------------------------------------------------
def pixelate(image, pixel_size=18):
    small = image.resize(
        (image.width // pixel_size, image.height // pixel_size),
        resample=Image.NEAREST
    )
    return small.resize(image.size, Image.NEAREST)

# --------------------------------------------------
# 2. Pencil Sketch (Pillow only)
# --------------------------------------------------
def pencil_sketch(image):
    gray = ImageOps.grayscale(image)
    inverted = ImageOps.invert(gray)

    blur = inverted.filter(ImageFilter.GaussianBlur(radius=25))
    blended = Image.blend(gray, blur, alpha=0.6)

    edges = blended.filter(ImageFilter.EDGE_ENHANCE_MORE)
    return edges.convert("RGB")

# --------------------------------------------------
# 3. Blur Background (Pillow only)
# --------------------------------------------------
def blur_background(image):
    blur = image.filter(ImageFilter.GaussianBlur(radius=25))

    mask = Image.new("L", image.size, 255)
    mask_draw = ImageDraw = __import__("ImageDraw", fromlist=[""]).ImageDraw(mask)

    w, h = image.size
    box_w = int(w * 0.55)
    box_h = int(h * 0.55)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    mask_draw.rectangle([x1, y1, x2, y2], fill=0)

    return Image.composite(image, blur, mask)

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4, _ = st.columns([2,1])

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate(img), caption="Pixelated Face", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
