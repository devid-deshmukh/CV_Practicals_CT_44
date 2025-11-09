import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page title and layout
st.set_page_config(page_title="Computer Vision App", layout="wide")

st.title("🎯 Our Computer Vision Application")
st.write("Upload an image to see both image filtering and facial expression analysis!")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Display original image
    st.image(image, caption="Original Image", use_column_width=True)

    st.write("## Processing Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Image Filtering ")

        # Apply filters from aim1.py
        avg_blur = cv2.blur(img_bgr, (5,5))
        gaussian_blur = cv2.GaussianBlur(img_bgr, (5,5), 0)
        median_blur = cv2.medianBlur(img_bgr, 5)
        bilateral_blur = cv2.bilateralFilter(img_bgr, 9, 75, 75)

        # Convert back to RGB for display
        avg_blur_rgb = cv2.cvtColor(avg_blur, cv2.COLOR_BGR2RGB)
        gaussian_blur_rgb = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB)
        median_blur_rgb = cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB)
        bilateral_blur_rgb = cv2.cvtColor(bilateral_blur, cv2.COLOR_BGR2RGB)

        # Display filtered images
        st.image(avg_blur_rgb, caption="Average Blur", use_column_width=True)
        st.image(gaussian_blur_rgb, caption="Gaussian Blur", use_column_width=True)
        st.image(median_blur_rgb, caption="Median Blur", use_column_width=True)
        st.image(bilateral_blur_rgb, caption="Bilateral Blur", use_column_width=True)

    with col2:
        st.subheader("😊 Facial Expression Analysis " \
        "")

        try:
            from deepface import DeepFace

            # Save temporary file for DeepFace
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                cv2.imwrite(tmp_file.name, img_bgr)
                tmp_path = tmp_file.name

            # Analyze facial expressions
            analysis = DeepFace.analyze(
                img_path=tmp_path,
                actions=['emotion'],
                enforce_detection=False
            )

            # Clean up temporary file
            os.unlink(tmp_path)

            # Ensure analysis is a list
            if isinstance(analysis, dict):
                analysis = [analysis]

            # Display results
            for idx, face in enumerate(analysis, start=1):
                st.write(f"**Face {idx}:**")
                st.write(f"Dominant Emotion: `{face['dominant_emotion']}`")

                # Show emotion probabilities as a bar chart
                emotions = face['emotion']
                st.bar_chart(emotions)

                # Draw bounding box and label
                region = face.get('region', {})
                x = int(region.get('x', 0))
                y = int(region.get('y', 0))
                w = int(region.get('w', 0))
                h = int(region.get('h', 0))

                if w > 0 and h > 0:
                    annotated_img = img_bgr.copy()
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(annotated_img, face['dominant_emotion'], (x, max(y - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Convert back to RGB for display
                    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="Detected Face with Emotion", use_column_width=True)

        except Exception as e:
            st.error(f"Error in facial analysis: {str(e)}")
            st.info("Make sure you have installed: `pip install deepface`")

else:
    st.info("👆 Please upload an image to get started")

st.markdown("---")
st.write("**Features:**")
st.write("- 🖼️ **Left**: Image filtering with 4 different blur techniques")
st.write("- 😊 **Right**: Facial expression analysis with emotion detection")