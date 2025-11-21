import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageOps
import mediapipe as mp

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")

st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#==============================
# FACE MESH SETUP
#==============================
mp_face_mesh = mp.solutions.face_mesh


#==============================
# 1. Pixelate ONLY the face
#==============================
def pixelate_face(img):
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    # Run face detection
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        results = fm.process(img_np)

        if not results.multi_face_landmarks:
            return img  # no face found → return original

        face = results.multi_face_landmarks[0]

        # Get bounding box of face from landmarks
        xs = [lm.x for lm in face.landmark]
        ys = [lm.y for lm in face.landmark]

        xmin = int(min(xs) * w)
        xmax = int(max(xs) * w)
        ymin = int(min(ys) * h)
        ymax = int(max(ys) * h)

        face_region = img.crop((xmin, ymin, xmax, ymax))

        # Pixelate
        small = face_region.resize((20, 20), Image.NEAREST)
        pixelated = small.resize(face_region.size, Image.NEAREST)

        # Paste back
        output = img.copy()
        output.paste(pixelated, (xmin, ymin))
        return output


#==============================
# 2. Hide Eyes with accurate detection
#==============================
def hide_eyes(img):
    w, h = img.size
    img_np = np.array(img)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        results = fm.process(img_np)

        if not results.multi_face_landmarks:
            return img

        face = results.multi_face_landmarks[0]

        LEFT_EYE_CENTER = 468
        RIGHT_EYE_CENTER = 473

        lx = int(face.landmark[LEFT_EYE_CENTER].x * w)
        ly = int(face.landmark[LEFT_EYE_CENTER].y * h)
        rx = int(face.landmark[RIGHT_EYE_CENTER].x * w)
        ry = int(face.landmark[RIGHT_EYE_CENTER].y * h)

        # Size relative to head size
        radius = int(min(w, h) * 0.04)

        output = img.copy()
        draw = ImageDraw.Draw(output)

        draw.ellipse((lx-radius, ly-radius, lx+radius, ly+radius), fill="black")
        draw.ellipse((rx-radius, ry-radius, rx+radius, ry+radius), fill="black")

        return output


#==============================
# 3. Pencil Sketch (PIL simulated)
#==============================
def pencil_sketch(img):
    gray = ImageOps.grayscale(img)
    inverted = ImageOps.invert(gray)
    blur = inverted.filter(ImageFilter.GaussianBlur(25))

    # dodge blend
    def dodge(a, b):
        result = b * 255 / (255 - a)
        result[result > 255] = 255
        return result.astype('uint8')

    sketch = dodge(np.array(gray), np.array(blur))
    sketch_img = Image.fromarray(sketch)

    return sketch_img


#==============================
# 4. Blur Background (keep face sharp)
#==============================
def blur_background(img):
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        results = fm.process(img_np)

        if not results.multi_face_landmarks:
            return img

        face = results.multi_face_landmarks[0]

        xs = [lm.x for lm in face.landmark]
        ys = [lm.y for lm in face.landmark]

        xmin = int(min(xs) * w)
        xmax = int(max(xs) * w)
        ymin = int(min(ys) * h)
        ymax = int(max(ys) * h)

        mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((xmin, ymin, xmax, ymax), fill=255)

        blurred = img.filter(ImageFilter.GaussianBlur(25))
        final = Image.composite(img, blurred, mask)
        return final


#==============================
# DISPLAY 5 RESULTS
#==============================
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    with col1:
        st.image(img, caption="Original", use_column_width=True)

    with col2:
        st.image(pixelate_face(img), caption="Pixelated Face", use_column_width=True)

    with col3:
        st.image(hide_eyes(img), caption="Eyes Hidden", use_column_width=True)

    with col4:
        st.image(pencil_sketch(img), caption="Pencil Sketch", use_column_width=True)

    with col5:
        st.image(blur_background(img), caption="Blur Background", use_column_width=True)

