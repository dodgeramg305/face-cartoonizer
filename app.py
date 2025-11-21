import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ---------------------------------
# A – Pixelate Face Only
# ---------------------------------
def pixelate_face_only(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    output = img.copy()

    for (x, y, w, h) in faces:
        face_roi = output[y:y+h, x:x+w]
        small = cv2.resize(face_roi, (20, 20), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        output[y:y+h, x:x+w] = pixelated

    return output

# ---------------------------------
# B – Blur Eyes Only
# ---------------------------------
def blur_eyes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    output = img.copy()

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (fx, fy, fw, fh) in faces:
        face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
        face_roi_color = output[fy:fy+fh, fx:fx+fw]

        eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.2, 3)

        for (ex, ey, ew, eh) in eyes:
            eye_section = face_roi_color[ey:ey+eh, ex:ex+ew]
            blurred_eye = cv2.GaussianBlur(eye_section, (51, 51), 0)
            face_roi_color[ey:ey+eh, ex:ex+ew] = blurred_eye

    return output

# ---------------------------------
# C – Pencil Sketch
# ---------------------------------
def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (31, 31), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

# ---------------------------------
# D – Blur Background
# ---------------------------------
def blur_background(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return img

    (x, y, w, h) = faces[0]

    blurred = cv2.GaussianBlur(img, (55, 55), 0)
    output = blurred.copy()
    output[y:y+h, x:x+w] = img[y:y+h, x:x+w]
    return output

# ---------------------------------
# DISPLAY RESULTS
# ---------------------------------
if uploaded:
    img = Image.open(uploaded)
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        pix = pixelate_face_only(img_bgr)
        st.image(cv2.cvtColor(pix, cv2.COLOR_BGR2RGB),
                 caption="Pixelated Face", use_column_width=True)

    with col3:
        blurred = blur_eyes(img_bgr)
        st.image(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB),
                 caption="Eyes Blurred", use_column_width=True)

    with col4:
        sketch = pencil_sketch(img_bgr)
        st.image(cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB),
                 caption="Pencil Sketch", use_column_width=True)

    with col5:
        blur_bg = blur_background(img_bgr)
        st.image(cv2.cvtColor(blur_bg, cv2.COLOR_BGR2RGB),
                 caption="Blur Background", use_column_width=True)
