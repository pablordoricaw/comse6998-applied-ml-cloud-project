import streamlit as st
import json
import numpy as np
from PIL import Image
from google.cloud import aiplatform
import google.auth
from google.auth.transport.requests import Request
import requests
from imagenet_labels import labels
import time

# Initialize Vertex AI
aiplatform.init(
    project="applied-ml-cloud-project",  # Your text project ID
    location="us-east1",
)

# Initialize endpoint
ENDPOINT_ID = "3495263901816193024"
LOCATION = "us-east1"
PROJECT_ID = "applied-ml-cloud-project"
PROJECT_NUMBER = "527470660206"
DEDICATED_URL = f"https://{ENDPOINT_ID}.{LOCATION}-{PROJECT_NUMBER}.prediction.vertexai.goog/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"
# endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

# Get Google credentials and access token
credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
credentials.refresh(Request())
access_token = credentials.token

# Streamlit UI
st.title("ResNet Image Classification (Triton/Vertex AI)")
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=150)

    if st.button("Classify"):
        # --- Image Preprocessing ---
        # Resize to 224x224
        img = img.resize((224, 224))

        # Modify the image preprocessing part to create a batch of 32 identical images
        img_array = np.asarray(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
        single_img = np.expand_dims(img_array, axis=0)  # (1, 3, 224, 224)

        # Create a batch of 32 copies of the same image
        img_batch = np.repeat(single_img, 32, axis=0)  # (32, 3, 224, 224)

        # --- Payload Construction ---
        payload = {
            "inputs": [
                {
                    "name": "input.1",  # Must match Triton config
                    "shape": list(img_batch.shape), # img_array.shape
                    "datatype": "FP32",
                    "data": img_batch.flatten().tolist(),
                }
            ]
        }

        # --- Prediction ---
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        # Add this right before your API call
        st.subheader("Performance Metrics")
        start_time = time.time()

        response = requests.post(DEDICATED_URL, headers=headers, data=json.dumps(payload))

        # Calculate and display the elapsed time
        end_time = time.time()
        api_response_time = end_time - start_time
        st.write(f"API Response Time: {api_response_time:.3f} seconds")

        results = json.loads(response.text)

        # st.write("Full response structure:")
        # st.json(results)

        # Process predictions
        if 'outputs' in results:
            # Get the data array
            output_data = results['outputs'][0]['data']
            
            # Get the first 1000 values (first image prediction)
            predictions = np.array(output_data[:1000])
            
            # Get the top prediction
            top_class_index = np.argmax(predictions)
            confidence = predictions[top_class_index]
            
            st.subheader("Top Prediction")
            st.write(f"Class index: {top_class_index}")
            st.write(f"Label Name: {labels[top_class_index]}")
            st.write(f"Confidence: {confidence:.4f}")
                        
            # Show top 5 predictions
            st.subheader("Top 5 Class Indices")
            top_indices = np.argsort(predictions)[-5:][::-1]
            for i, idx in enumerate(top_indices):
                st.write(f"{i+1}. Class Index: {idx}, Name: {labels[idx]}, Confidence: {predictions[idx]:.4f}")