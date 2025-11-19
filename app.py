import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")
st.title("🎨 Face Fun Factory – 5 Transformations + Bunny Ears")

# Load bunny ears
BUNNY_PATH = "assets/bunny_ears.png"
bunny = Image.open(BUNNY_PATH)
bunny = np.array(bunny)

# File upload
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.success("Image uploaded successfully!")

    img = np.array(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 1. FACE DETECTION USING HAAR
    # -----------------------------
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    faces = face_cascade.detectMultiScale(img_rgb, 1.3, 5)

    # -----------------------------
    # 2. PIXELATION
    # -----------------------------
    def pixelate(image, size=15):
        h, w = image.shape[:2]
        temp = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

    pixelated = img_rgb.copy()
    for (x, y, w, h) in faces:
        face = pixelated[y:y+h, x:x+w]
        face = pixelate(face, size=20)
        pixelated[y:y+h, x:x+w] = face

    # -----------------------------
    # 3. BLUE FACE FILTER
    # -----------------------------
    blue = img_rgb.copy()
    for (x, y, w, h) in faces:
        blue[y:y+h, x:x+w, :] = [50, 80, 200]  # blue

    # -----------------------------
    # 4. BLACK CIRCLES OVER EYES
    # -----------------------------
    eye_mask = img_rgb.copy()
    for (x, y, w, h) in faces:
        roi_color = eye_mask[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_color)

        for (ex, ey, ew, eh) in eyes:
            center = (x + ex + ew//2, y + ey + eh//2)
            radius = int(ew * 0.7)
            cv2.circle(eye_mask, center, radius, (0, 0, 0), -1)

    # -----------------------------
    # 5. CARTOONIZER
    # -----------------------------
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9, 9
    )

    color = cv2.bilateralFilter(img_rgb, 9, 200, 200)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # -----------------------------
    # ADD BUNNY EARS
    # -----------------------------
    bunny_output = cartoon.copy()

    for (x, y, w, h) in faces:
        # Resize bunny ears relative to face width
        ears_w = int(w * 1.2)
        ears_h = int(ears_w * (bunny.shape[0] / bunny.shape[1]))

        resized_ears = cv2.resize(bunny, (ears_w, ears_h))

        # Overlay position (above the face)
        pos_x = x - int((ears_w - w) / 2)
        pos_y = y - ears_h + 20

        for i in range(ears_h):
            for j in range(ears_w):
                if 0 <= pos_y + i < bunny_output.shape[0] and 0 <= pos_x + j < bunny_output.shape[1]:
                    pixel = resized_ears[i, j]
                    if pixel[3] > 10:  # respect transparency
                        bunny_output[pos_y + i, pos_x + j] = pixel[:3]

    # -----------------------------
    # DISPLAY 5 PANELS
    # -----------------------------
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)

    col1.image(img_rgb, caption="Original", use_column_width=True)
    col2.image(pixelated, caption="Pixelated Face", use_column_width=True)
    col3.image(blue, caption="Blue Face Filter", use_column_width=True)
    col4.image(eye_mask, caption="Eyes Hidden", use_column_width=True)
    col5.image(bunny_output, caption="Cartoon + Bunny Ears", use_column_width=True)
