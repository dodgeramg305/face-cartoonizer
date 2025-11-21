import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# -----------------------------------------------------------------------------
# Pixelate CENTER REGION (simulated “face pixelation”)
# -----------------------------------------------------------------------------
def pixelate_center(img, box_ratio=0.55, pixel_scale=16):
    w, h = img.size

    # Define centered square region (simulated face box)
    box_size = int(min(w, h) * box_ratio)
    left = (w - box_size) // 2
    top = (h - box_size) // 2
    right = left + box_size
    bottom = top + box_size

    face_region = img.crop((left, top, right, bottom))

    # Pixelate the cropped region
    small = face_region.resize((box_size // pixel_scale, box_size // pixel_scale), Image.BILINEAR)
    pixelated = small.resize((box_size, box_size), Image.NEAREST)

    # Paste pixelated region back
    result = img.copy()
    result.paste(pixelated, (left, top))
    return result


# -----------------------------------------------------------------------------
# Pencil Sketch (light, not dark)
# -----------------------------------------------------------------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(25))

    # Light sketch
    dodge = ImageOps.blend(gray, blur, 0.2)

    # Enhance lines
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    sketch = Image.blend(dodge, edges, 0.35)
    return sketch.convert("RGB")


# -----------------------------------------------------------------------------
# Blur Background but keep same image size
# -----------------------------------------------------------------------------
def blur_background(img):
    w, h = img.size

    # Blur whole image
    blurred = img.filter(ImageFilter.GaussianBlur(30))

    # Circular mask
    mask = Image.new("L", (w, h), 0)
    cx, cy = w // 2, h // 2
    r = int(min(w, h) * 0.45)

    for x in range(w):
        for y in range(h):
            if (x - cx)**2 + (y - cy)**2 < r*r:
                mask.putpixel((x, y), 255)

    # Combine: face area stays sharp, background blurred
    result = Image.composite(img, blurred, mask)
    return result


# -----------------------------------------------------------------------------
# DISPLAY OUTPUT
# -----------------------------------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4 = st.columns(1)[0]

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_center(img), caption="Pixelated Face (Center)", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
