import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import mediapipe as mp

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

mp_face = mp.solutions.face_mesh


# -----------------------------------
# PIXELATE ONLY THE FACE (soft edges)
# -----------------------------------
def pixelate_face(img, pixel_size=24):
    np_img = np.array(img)
    h, w, _ = np_img.shape

    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(np_img)

        if not results.multi_face_landmarks:
            return img  # No face detected

        landmarks = results.multi_face_landmarks[0]

        # ▪ Extract face bounding box from all 468 points
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]

        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))

        # Crop face
        face = img.crop((x1, y1, x2, y2))

        # Pixelate face only
        face_small = face.resize((face.width // pixel_size, face.height // pixel_size), Image.NEAREST)
        face_pixelated = face_small.resize(face.size, Image.NEAREST)

        # Paste back onto original
        img_copy = img.copy()
        img_copy.paste(face_pixelated, (x1, y1))

        return img_copy


# ---------------------
# Pencil Sketch (clean)
# ---------------------
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(radius=15))
    blend = ImageOps.colorize(ImageOps.invert(ImageOps.blend(gray, blur, 0.5)),
                               black="black", white="white")
    return blend


# ---------------------------------------
# Blur Background (same size as others)
# ---------------------------------------
def blur_background(img):
    W, H = img.size
    output_size = (600, 600)

    bg = img.resize(output_size).filter(ImageFilter.GaussianBlur(radius=20))
    face = img.resize(output_size)

    # simple circular mask
    mask = Image.new("L", output_size, 0)
    r = int(output_size[0] * 0.35)
    cx, cy = output_size[0] // 2, output_size[1] // 2

    for x in range(output_size[0]):
        for y in range(output_size[1]):
            if (x - cx)**2 + (y - cy)**2 < r*r:
                mask.putpixel((x, y), 255)

    return Image.composite(face, bg, mask)


# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    c1, c2, c3 = st.columns(3)
    c4 = st.columns(1)[0]

    with c1:
        st.image(img, caption="Original", use_column_width=True)

    with c2:
        st.image(pixelate_face(img), caption="Pixelated Face Only", use_column_width=True)

    with c3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with c4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
