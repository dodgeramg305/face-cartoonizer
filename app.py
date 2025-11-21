import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------
# Pixelate FACE ONLY
# -----------------------
def pixelate_face(img, scale=16):
    w, h = img.size

    # Convert to NumPy
    np_img = np.array(img)

    # Assume center face for simple version
    cx, cy = w // 2, h // 2
    box_w, box_h = int(w * 0.45), int(h * 0.45)

    x1 = cx - box_w // 2
    y1 = cy - box_h // 2
    x2 = cx + box_w // 2
    y2 = cy + box_h // 2

    face_crop = img.crop((x1, y1, x2, y2))

    small = face_crop.resize((face_crop.width // scale, face_crop.height // scale), Image.NEAREST)
    pixelated = small.resize(face_crop.size, Image.NEAREST)

    img_copy = img.copy()
    img_copy.paste(pixelated, (x1, y1))

    return img_copy

# -----------------------
# Pencil Sketch (bright)
# -----------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(radius=18))

    # Bright sketch
    dodge = ImageOps.blend(gray, blur, 0.2)

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    final = Image.blend(dodge, edges, 0.35)
    return final.convert("RGB")

# -----------------------
# Blur Background Circle
# -----------------------
def blur_background(img):
    output_size = (350, 350)

    blurred = img.filter(ImageFilter.GaussianBlur(radius=22)).resize(output_size)

    circle_mask = Image.new("L", output_size, 0)

    cx, cy = output_size[0] // 2, output_size[1] // 2
    r = int(output_size[0] * 0.38)

    for x in range(output_size[0]):
        for y in range(output_size[1]):
            if (x - cx)**2 + (y - cy)**2 < r*r:
                circle_mask.putpixel((x, y), 255)

    face_resized = img.resize(output_size)

    return Image.composite(face_resized, blurred, circle_mask)

# -------------------------------------------------------------------
# DISPLAY RESULTS — 4 SMALL BOXES EXACTLY LIKE YOUR SCREENSHOT
# -------------------------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face(img), caption="Pixelated Face Only", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
