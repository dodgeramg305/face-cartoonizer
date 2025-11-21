import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------
# Pixelation (soft version)
# -----------------------
def pixelate(img, scale=32):
    w, h = img.size
    img_small = img.resize((w // scale, h // scale), resample=Image.BILINEAR)
    return img_small.resize((w, h), Image.NEAREST)

# -----------------------
# Pencil Sketch (stronger edges)
# -----------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(radius=18))
    
    # dodge blend (sketch effect)
    blended = ImageOps.dodge(gray, blur)
    
    # increase edges for more “sketch” feel
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    
    final = Image.blend(blended, edges, alpha=0.45)
    return final.convert("RGB")

# -----------------------
# Blur Background (proper square + centered)
# -----------------------
def blur_background(img):
    w, h = img.size

    # SIZE MATCHES ALL OTHER GRID IMAGES
    output_size = (600, 600)

    # blurred copy
    blurred = img.filter(ImageFilter.GaussianBlur(radius=22)).resize(output_size)

    # mask circle in center
    mask = Image.new("L", output_size, 0)
    circle = Image.new("L", output_size, 0)

    # circle radius = 35% of width
    r = int(output_size[0] * 0.35)
    cx, cy = output_size[0] // 2, output_size[1] // 2

    for x in range(output_size[0]):
        for y in range(output_size[1]):
            if (x - cx)**2 + (y - cy)**2 < r*r:
                circle.putpixel((x, y), 255)

    mask = circle

    # resized original face centered
    face = img.resize((output_size[0], output_size[1]))

    # combine
    final = Image.composite(face, blurred, mask)
    return final

# -----------------------------------------------------
# DISPLAY RESULTS
# -----------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4 = st.columns(1)[0]

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate(img), caption="Pixelated Face", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
