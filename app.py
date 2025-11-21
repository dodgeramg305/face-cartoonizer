import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -------------------------------------------------------
# Unified display size (MATCHES blur background box size)
# -------------------------------------------------------
DISPLAY_SIZE = (600, 600)

def resize_for_display(img):
    return img.resize(DISPLAY_SIZE)

# -----------------------
# Pixelate ONLY the face
# -----------------------
def pixelate_face_only(img):
    img_gray = ImageOps.grayscale(img)
    arr = np.array(img_gray)

    # face-like prior using center area
    h, w = arr.shape
    cx, cy = w // 2, h // 2
    box_w, box_h = w // 3, h // 3

    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2

    face_region = img.crop((x1, y1, x2, y2))

    # pixelate
    face_small = face_region.resize((20, 20), Image.NEAREST)
    pixelated_face = face_small.resize(face_region.size, Image.NEAREST)

    # paste back
    result = img.copy()
    result.paste(pixelated_face, (x1, y1))
    return result

# -----------------------
# Pencil Sketch (lighter)
# -----------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(18))

    dodge = ImageOps.blend(gray, blur, 0.2)

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)

    final = Image.blend(dodge, edges, 0.25)
    return final.convert("RGB")

# -----------------------
# Blur Background
# -----------------------
def blur_background(img):
    blurred = img.filter(ImageFilter.GaussianBlur(radius=25)).resize(DISPLAY_SIZE)

    mask = Image.new("L", DISPLAY_SIZE, 0)
    r = int(DISPLAY_SIZE[0] * 0.35)
    cx, cy = DISPLAY_SIZE[0] // 2, DISPLAY_SIZE[1] // 2

    for x in range(DISPLAY_SIZE[0]):
        for y in range(DISPLAY_SIZE[1]):
            if (x - cx)**2 + (y - cy)**2 < r*r:
                mask.putpixel((x, y), 255)

    face = img.resize(DISPLAY_SIZE)
    final = Image.composite(face, blurred, mask)
    return final


# -----------------------------------------------------
# DISPLAY RESULTS (ALL BOXES SAME SIZE)
# -----------------------------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image(resize_for_display(img), caption="Original", use_column_width=True)

    with col2:
        st.image(resize_for_display(pixelate_face_only(img)), caption="Pixelated Face Only", use_column_width=True)

    with col3:
        st.image(resize_for_display(pencil_sketch(img)), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
