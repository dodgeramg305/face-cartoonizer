import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


# ==========================
# FACE PIXELATION (ONLY FACE)
# ==========================
def pixelate_face(img_pil, scale=18):
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Haar cascade (built into OpenCV)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return img_pil  # no face found

    for (x, y, w, h) in faces:
        face_region = img[y:y+h, x:x+w]

        # pixelate only this rectangle
        small = cv2.resize(face_region, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        img[y:y+h, x:x+w] = pixelated

    return Image.fromarray(img)


# ==========================
# PENCIL SKETCH
# ==========================
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inv = ImageOps.invert(gray)
    blur = inv.filter(ImageFilter.GaussianBlur(radius=18))

    # Safe dodge formula
    gray_np = np.array(gray).astype('float')
    blur_np = np.array(blur).astype('float')

    result = np.minimum(255, (gray_np * 255) / (255 - blur_np + 1e-4))
    result = result.astype("uint8")

    return Image.fromarray(result)


# ==========================
# BLUR BACKGROUND (SAME BOX SIZE)
# ==========================
def blur_background(img):
    output_size = (400, 400)  # same size as other images
    base = img.resize(output_size)

    blurred = base.filter(ImageFilter.GaussianBlur(radius=22))

    mask = Image.new("L", output_size, 0)
    cx, cy = output_size[0]//2, output_size[1]//2
    r = int(output_size[0] * 0.40)

    for x in range(output_size[0]):
        for y in range(output_size[1]):
            if (x - cx)**2 + (y - cy)**2 < r*r:
                mask.putpixel((x, y), 255)

    return Image.composite(base, blurred, mask)


# ==========================
# DISPLAY RESULTS
# ==========================
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4 = st.columns(1)[0]

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face(img), caption="Pixelated Face (Face Only)", use_column_width=True)

    with col3:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col4:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)
