import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")
st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -----------------------------
# Function: Pixelate (softer)
# -----------------------------
def pixelate(img, scale=20):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

# -----------------------------
# Function: Pencil Sketch
# -----------------------------
def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (31, 31), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

# -----------------------------
# Function: Hide Eyes (smaller circles)
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

def hide_eyes(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        res = fm.process(rgb)
        if not res.multi_face_landmarks:
            return img

        face = res.multi_face_landmarks[0]

        left_pts = [face.landmark[i] for i in [33, 133]]
        right_pts = [face.landmark[i] for i in [362, 263]]

        def px(pt): return int(pt.x * w), int(pt.y * h)

        lx1, ly1 = px(left_pts[0])
        lx2, ly2 = px(left_pts[1])
        rx1, ry1 = px(right_pts[0])
        rx2, ry2 = px(right_pts[1])

        # Smaller radius
        left_r = int(np.linalg.norm([lx1-lx2, ly1-ly2]) * 0.9)
        right_r = int(np.linalg.norm([rx1-rx2, ry1-ry2]) * 0.9)

        out = img.copy()
        cv2.circle(out, (lx1, ly1), left_r, (0, 0, 0), -1)
        cv2.circle(out, (rx1, ry1), right_r, (0, 0, 0), -1)
        return out

# -----------------------------
# Display Output
# -----------------------------
if uploaded:
    img = Image.open(uploaded)
    img = np.array(img)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    c1, c2, c3 = st.columns(3)
    c4, c5 = st.columns(2)

    with c1:
        st.image(img, caption="Original", use_column_width=True)

    with c2:
        pixel = pixelate(img_bgr, scale=35)
        st.image(cv2.cvtColor(pixel, cv2.COLOR_BGR2RGB),
                 caption="Pixelated Face", use_column_width=True)

    with c3:
        hidden = hide_eyes(img_bgr)
        st.image(cv2.cvtColor(hidden, cv2.COLOR_BGR2RGB),
                 caption="Eyes Hidden", use_column_width=True)

    with c4:
        sketch = pencil_sketch(img_bgr)
        st.image(cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB),
                 caption="Pencil Sketch", use_column_width=True)
