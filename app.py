import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

# ---------------------------
# App Config & Intro
# ---------------------------
st.set_page_config(page_title="Face Cartoonizer", layout="wide")

st.title("🎨 Face Cartoonizer – Facial Feature Aware")
st.write(
    """
Upload a photo and this app will:

1. **Detect faces** using MediaPipe's face detection.  
2. **Cartoonize** either:
   - the **entire image**, or  
   - **only the detected faces** (background stays normal).  

This connects to **Facial Recognition / Facial Feature Recognition** by:
- Automatically locating faces (bounding boxes).
- Applying a privacy-friendly and fun cartoon filter to face regions.
"""
)

# ---------------------------
# MediaPipe Face Detection Setup
# ---------------------------
mp_face_detection = mp.solutions.face_detection


def detect_faces_mediapipe(img_rgb, min_conf=0.5):
    """
    Detect faces in an RGB image using MediaPipe.
    Returns:
        - list of bounding boxes: [(x1, y1, x2, y2), ...]
        - annotated RGB image with rectangles drawn.
    """
    h, w, _ = img_rgb.shape
    bboxes = []
    annotated = img_rgb.copy()

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=min_conf
    ) as face_detector:
        results = face_detector.process(img_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                w_box = int(bboxC.width * w)
                h_box = int(bboxC.height * h)

                # Clamp values to image boundaries
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x + w_box)
                y2 = min(h, y + h_box)

                if x2 > x and y2 > y:
                    bboxes.append((x, y, x2, y2))
                    cv2.rectangle(annotated, (x, y), (x2, y2), (0, 255, 0), 2)

    return bboxes, annotated


# ---------------------------
# Cartoon Effect
# ---------------------------
def cartoonize_bgr(
    img_bgr, edge_detail_level=3, smooth_strength=3, bilateral_filter_size=9
):
    """
    Apply a cartoon effect to a BGR image.
    - edge_detail_level (1-5): controls edge block size (fine vs bold lines)
    - smooth_strength (1-5): number of bilateral filter passes
    """
    # Convert to gray and blur for better edge detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)

    # Ensure block size is odd and reasonably sized
    block_size = int(edge_detail_level) * 2 + 1  # 3,5,7,9,11 ...
    block_size = max(3, block_size)

    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        2,
    )

    # Smooth the color image but keep edges
    color = img_bgr.copy()
    smooth_strength = int(smooth_strength)
    smooth_strength = max(1, min(smooth_strength, 5))
    for _ in range(smooth_strength):
        color = cv2.bilateralFilter(
            color, d=bilateral_filter_size, sigmaColor=75, sigmaSpace=75
        )

    # Combine color image with edges
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def process_image(img_rgb, mode, edge_detail_level, smooth_strength):
    """
    Main processing pipeline:
    - Detect faces
    - Apply cartoon filter based on selected mode
    Returns:
        cartoonized_rgb (np.array),
        annotated_faces_rgb (np.array)
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Detect faces on RGB image
    bboxes, annotated = detect_faces_mediapipe(img_rgb)

    if mode == "Cartoonize whole image":
        cartoon_bgr = cartoonize_bgr(
            img_bgr,
            edge_detail_level=edge_detail_level,
            smooth_strength=smooth_strength,
        )
        cartoon_rgb = cv2.cvtColor(cartoon_bgr, cv2.COLOR_BGR2RGB)
        return cartoon_rgb, annotated

    elif mode == "Cartoonize faces only":
        output_bgr = img_bgr.copy()

        # If no faces detected, fall back to full-image cartoon
        if not bboxes:
            cartoon_bgr = cartoonize_bgr(
                img_bgr,
                edge_detail_level=edge_detail_level,
                smooth_strength=smooth_strength,
            )
            cartoon_rgb = cv2.cvtColor(cartoon_bgr, cv2.COLOR_BGR2RGB)
            return cartoon_rgb, annotated

        # Apply cartoon effect only inside face bounding boxes
        for (x1, y1, x2, y2) in bboxes:
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            cartoon_roi = cartoonize_bgr(
                roi,
                edge_detail_level=edge_detail_level,
                smooth_strength=smooth_strength,
            )
            output_bgr[y1:y2, x1:x2] = cartoon_roi

        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        return output_rgb, annotated

    else:
        # fallback (should not happen)
        return img_rgb, annotated


# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.header("Settings")

mode = st.sidebar.radio(
    "Cartoonization Mode",
    ["Cartoonize whole image", "Cartoonize faces only"],
)

edge_detail_level = st.sidebar.slider(
    "Edge detail (1 = bold lines, 5 = fine lines)", min_value=1, max_value=5, value=3
)

smooth_strength = st.sidebar.slider(
    "Color smoothing strength (1–5)", min_value=1, max_value=5, value=3
)

st.sidebar.markdown(
    """
**Tips:**
- Use a clear, front-facing photo.
- Try increasing *Edge detail* for more line-art style.
- Increase *Color smoothing* for a more "anime"/cartoon look.
"""
)

# ---------------------------
# File Uploader
# ---------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    cartoon_rgb, annotated_faces = process_image(
        img_rgb,
        mode=mode,
        edge_detail_level=edge_detail_level,
        smooth_strength=smooth_strength,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(img_rgb, use_column_width=True)

    with col2:
        st.subheader("Cartoonized Image")
        st.image(cartoon_rgb, use_column_width=True)

    with st.expander("Show detected faces (bounding boxes)"):
        st.image(annotated_faces, caption="Faces detected", use_column_width=True)

else:
    st.info("👆 Upload an image to get started.")
