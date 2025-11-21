import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# ----------------------------------------------------------
# FACE ESTIMATION (simple center-face crop, PIL-compatible)
# ----------------------------------------------------------
def get_face_box(img):
    """
    Very simple heuristic: assume the face is in the center 40% of the image.
    Works for portrait photos where the user uploads a normal selfie/headshot.
    """
    w, h = img.size
    box_w, box_h = int(w * 0.45), int(h * 0.45)

    left = (w - box_w) // 2
    top = (h - box_h) // 2
    right = left + box_w
    bottom = top + box_h

    return (left, top, right, bottom)


# ----------------------------------------------------------
# PIXELATE ONLY THE FACE
# ----------------------------------------------------------
def pixelate_face_only(img, pixel_scale=12):
    img = img.copy()
    face_box = get_face_box(img)

    face = img.crop(face_box)
    fw, fh = face.size

    # pixelate
    small = face.resize((fw // pixel_scale, fh // pixel_scale), Image.NEAREST)
    pixelated = small.resize((fw, fh), Image.NEAREST)

    # paste back
    img.paste(pixelated, face_box)
    return img


# ----------------------------------------------------------
# PENCIL SKETCH — balanced, not too dark
# ----------------------------------------------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)

    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(radius=25))

    # Dodge blend
    dodge = ImageOps.blend(gray, blur, 0.2)

    # Edge enhancement for real sketch feeling
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    final = Image.blend(dodge, edges, 0.35)
    return final.convert("RGB")


# ----------------------------------------------------------
# BLUR BACKGROUND (square output)
# ----------------------------------------------------------
def blur_background(img):
    output_size = (500, 500)

    # Resize original
    face = img.resize(output_size)

    # Blur copy
    blurred = img.filter(ImageFilter.GaussianBlur(radius=30)).resize(output_size)

    # Create centered circle mask
    mask = Image.new("L", output_size, 0)
    cx, cy = output_size[0] // 2, output_size[1] // 2
    r = int(output_size[0] * 0.38)

    for x in range(output_size[0]):
        for y in range(output_size[1]):
            if (x - cx)**2 + (y - cy)**2 <= r*r:
                mask.putpixel((x, y), 255)

    final = Image.composite(face, blurred, mask)
    return final


# ----------------------------------------------------------
# DISPLAY GRID
# ----------------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4 = st.columns(1)[0]

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face_only(img), caption="Pixelated Face Only", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
