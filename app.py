import os
import gc
import io
import cv2
import numpy as np
import requests
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("Bone Fracture Detection with Deep Learning")

st.markdown(
    """
    Hi, this is Jeyanth Kannan!  
    This app uses deep learning (YOLO models) to detect bone fractures in X-ray images.  
    Upload your image or test with sample images to explore its capabilities.
    """
)

# -------------------------
# Step 1: Load Models
# -------------------------

@st.cache_resource
def load_models():
    model_files = ["model1.pt", "model3.pt"]

    # Check for existence
    for file in model_files:
        if not os.path.exists(file):
            st.error(f"Missing model file: {file}")
            st.stop()

    models = [YOLO(file) for file in model_files]
    for model in models:
        model.conf = 0.4
    return models

models = load_models()

# -------------------------
# Step 2: Resize Image
# -------------------------

def resize_image(image, max_width=400, max_height=400):
    width, height = image.size
    if width > max_width:
        new_width = max_width
        new_height = int((max_width / width) * height)
        image = image.resize((new_width, new_height))
    if height > max_height:
        new_height = max_height
        new_width = int((max_height / height) * width)
        image = image.resize((new_width, new_height))
    return image

# -------------------------
# Step 3: Handle File Upload
# -------------------------

def handle_file_upload(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        return image
    except Exception as e:
        st.error(f"Error in file upload: {e}")
        raise

# -------------------------
# Step 4: Fracture Detection
# -------------------------

def detect_fracture(image):
    try:
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
        results = [model(image_cv) for model in models]
        fracture_detected = any(len(result) > 0 and len(result[0].boxes) > 0 for result in results)

        if fracture_detected:
            best_result = max(results, key=lambda x: max((box.conf.item() for box in x[0].boxes), default=0))
            boxes = best_result[0].boxes.xyxy.cpu().numpy()
            confidences = best_result[0].boxes.conf.cpu().numpy()

            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

            annotated_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            return True, annotated_image
        else:
            return False, None

    except Exception as e:
        st.error(f"Error during detection: {e}")
        return False, None
    finally:
        gc.collect()

# -------------------------
# Step 5: Main App
# -------------------------

def main():
    st.sidebar.header("Options")
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            image = handle_file_upload(uploaded_file)
            resized_image = resize_image(image)
            col1, col2 = st.columns(2)

            with col1:
                st.image(resized_image, caption="Uploaded Image")

            if st.button("Detect Fracture"):
                with st.spinner("Detecting fractures... 🔍"):
                    fracture_detected, annotated_image = detect_fracture(image)

                if fracture_detected:
                    st.success("Fracture detected!")
                    resized_annotated_image = resize_image(annotated_image)
                    with col2:
                        st.image(resized_annotated_image, caption="Detected Image")
                else:
                    st.warning("No fractures detected.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
