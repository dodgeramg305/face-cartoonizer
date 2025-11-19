import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Interactive Face Effects Studio", layout="wide")

st.title("🎭 Interactive Face Effects Studio")
st.write(
    """
This app detects faces in an uploaded image and lets you apply different **face-only effects**:

- Blur faces  
- Pixelate faces  
- Cartoonize faces  
- Add black bars over the eyes (privacy mode)  

It uses **OpenCV's Haar cascade face detector**, which is related to what we covered in class (face detection & privacy filters).
"""
)

# ---------------------------
# Helper: Convert PIL <-> OpenCV
# ---------------------------
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def bgr_to_pil(bgr_img: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# ---------------------------
# Face Detection (Haar Cascade)
# ---------------------------
@st.cache_resource
def load_face_detector():
    # Use OpenCV's built-in path for Haar cascades
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

def detect_faces(img_bgr, scaleFactor=1.1, minNeighbors=5):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = load_face_detector()
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(40, 40)
    )
    return faces  # list of (x, y, w, h)

# ---------------------------
# Effects
# ---------------------------
def apply_blur(img_bgr, faces, ksize=35):
    out = img_bgr.copy()
    for (x, y, w, h) in faces:
        face_roi = out[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        blurred = cv2.GaussianBlur(face_roi, (ksize, ksize), 0)
        out[y:y+h, x:x+w] = blurred
    return out

def apply_pixelate(img_bgr, faces, blocks=10):
    out = img_bgr.copy()
    for (x, y, w, h) in faces:
        face_roi = out[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue
        # Downscale then upscale to create pixelation
        small = cv2.resize(face_roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        out[y:y+h, x:x+w] = pixelated
    return out

def apply_cartoon(img_bgr, faces=None, whole_image=False):
    out = img_bgr.copy()

    def cartoonize_region(region_bgr):
        gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            9
        )
        color = cv2.bilateralFilter(region_bgr, 9, 200, 200)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    if whole_image or not faces:
        return cartoonize_region(out)

    for (x, y, w, h) in faces:
        roi = out[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        out[y:y+h, x:x+w] = cartoonize_region(roi)
    return out

def apply_black_bar(img_bgr, faces, thickness_ratio=0.18):
    out = img_bgr.copy()
    for (x, y, w, h) in faces:
        bar_h = int(h * thickness_ratio)
        bar_y1 = y + int(h * 0.3)
        bar_y2 = min(y + h, bar_y1 + bar_h)
        cv2.rectangle(out, (x, bar_y1), (x + w, bar_y2), (0, 0, 0), -1)
    return out

def draw_face_boxes(img_bgr, faces, color=(0, 255, 0), thickness=2):
    out = img_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Controls")

effect_choices = st.sidebar.multiselect(
    "Select face effects to apply",
    ["Blur faces", "Pixelate faces", "Cartoonize faces", "Black bar over eyes"],
    default=["Cartoonize faces"]
)

face_mode = st.sidebar.radio(
    "Cartoon mode",
    ["Faces only", "Whole image (cartoon)"],
    index=0
)

blur_intensity = st.sidebar.slider(
    "Blur strength (for 'Blur faces')",
    min_value=15,
    max_value=75,
    value=35,
    step=2
)

pixel_blocks = st.sidebar.slider(
    "Pixelation level (for 'Pixelate faces')",
    min_value=4,
    max_value=40,
    value=12
)

show_boxes = st.sidebar.checkbox("Show face detection boxes preview", value=True)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ---------------------------
# Main App Logic
# ---------------------------
if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img_bgr = pil_to_bgr(pil_img)

    # Detect faces
    faces = detect_faces(img_bgr)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(pil_img, use_container_width=True)
        st.write(f"Detected **{len(faces)}** face(s).")

    # Start from original for processing
    processed = img_bgr.copy()

    # If "Cartoonize faces" selected and face_mode is "Whole image", do that first
    if "Cartoonize faces" in effect_choices and face_mode == "Whole image":
        processed = apply_cartoon(processed, faces=None, whole_image=True)

    # Apply effects to faces only
    if faces is not None and len(faces) > 0:
        if "Blur faces" in effect_choices:
            processed = apply_blur(processed, faces, ksize=blur_intensity)

        if "Pixelate faces" in effect_choices:
            processed = apply_pixelate(processed, faces, blocks=pixel_blocks)

        if "Cartoonize faces" in effect_choices and face_mode == "Faces only":
            processed = apply_cartoon(processed, faces=faces, whole_image=False)

        if "Black bar over eyes" in effect_choices:
            processed = apply_black_bar(processed, faces)

    processed_pil = bgr_to_pil(processed)

    with col2:
        st.subheader("Processed Image")
        st.image(processed_pil, use_container_width=True)

    if show_boxes and faces is not None and len(faces) > 0:
        boxed = draw_face_boxes(img_bgr, faces)
        st.subheader("Face Detection Preview")
        st.image(bgr_to_pil(boxed), use_container_width=True)

else:
    st.info("👆 Upload an image to get started.")
