import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="Face Fun Factory – Transformations", layout="wide")

st.title("Face Fun Factory – Transformations")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ---------------------------------------------------
# FUNCTION: Mild Pixelation (Option A)
# ---------------------------------------------------
def pixelate(img, pixel_size=25):
    h, w = img.shape[:2]

    # shrink image
    small = cv2.resize(img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)

    # grow back blocky
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------
# FUNCTION: Pencil Sketch
# ---------------------------------------------------
def pencil_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (31, 31), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------
# FUNCTION: Hide Eyes using FaceMesh
# ---------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh

def hide_eyes(img):

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Load Mediapipe model
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return img  # No face detected

        face = results.multi_face_landmarks[0]

        # Key eye points (left + right)
        left_eye_idxs = [33, 133]
        right_eye_idxs = [362, 263]

        # Convert to pixel positions
        def to_px(lm):
            return int(lm.x * w), int(lm.y * h)

        # Left eye
        lx1, ly1 = to_px(face.landmark[left_eye_idxs[0]])
        lx2, ly2 = to_px(face.landmark[left_eye_idxs[1]])

        # Right eye
        rx1, ry1 = to_px(face.landmark[right_eye_idxs[0]])
        rx2, ry2 = to_px(face.landmark[right_eye_idxs[1]])

        # Radius proportional to eye size — MUCH smaller than before
        left_r = int(np.linalg.norm([lx1 - lx2, ly1 - ly2]) * 0.9)
        right_r = int(np.linalg.norm([rx1 - rx2, ry1 - ry2]) * 0.9)

        output = img.copy()

        # Draw black circles (small, clean)
        cv2.circle(output, (lx1, ly1), left_r, (0, 0, 0), -1)
        cv2.circle(output, (rx1, ry1), right_r, (0, 0, 0), -1)

        return output


# ---------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------
if uploaded:

    # PIL → numpy
    img = Image.open(uploaded)
    img = np.array(img)

    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    # --- Original
    with col1:
        st.image(img, caption="Original", use_column_width=True)

    # --- Pixelated (mild)
    with col2:
        pix = pixelate(img_bgr)
        st.image(cv2.cvtColor(pix, cv2.COLOR_BGR2RGB),
                 caption="Pixelated Face", use_column_width=True)

    # --- Eyes Hidden
    with col3:
        hidden = hide_eyes(img_bgr)
        st.image(cv2.cvtColor(hidden, cv2.COLOR_BGR2RGB),
                 caption="Eyes Hidden", use_column_width=True)

    # --- Pencil Sketch
    with col4:
        sketch = pencil_sketch(img_bgr)
        st.image(cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB),
                 caption="Pencil Sketch", use_column_width=True)
