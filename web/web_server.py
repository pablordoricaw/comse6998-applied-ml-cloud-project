import streamlit as st
import json
import numpy as np
from PIL import Image
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(
    project="applied-ml-cloud-project",  # Your text project ID
    location="us-east1",
)

# Initialize endpoint
ENDPOINT_ID = "3495263901816193024"
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

# Streamlit UI
st.title("ResNet Image Classification (Triton/Vertex AI)")
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    if st.button("Classify"):
        # --- Image Preprocessing ---
        # Resize to 224x224
        img = img.resize((224, 224))

        # Convert to array and normalize
        img_array = np.asarray(img).astype(np.float32) / 255.0

        # ResNet expects CHW format (3, 224, 224)
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW

        # Add batch dimension (1, 3, 224, 224)
        img_array = np.expand_dims(img_array, axis=0)

        # --- Payload Construction ---
        payload = {
            "inputs": [
                {
                    "name": "INPUT_0",  # Must match Triton config
                    "shape": img_array.shape,
                    "datatype": "FP32",
                    "data": img_array.flatten().tolist(),
                }
            ]
        }

        # --- Prediction ---

        response = endpoint.raw_predict(
            body=json.dumps(payload), headers={"Content-Type": "application/json"}
        )

        # Process output (assuming 1000-class ImageNet output)
        results = json.loads(response.text)
        st.write(results.keys())
        outputs = np.array(results["outputs:"][0]["data"])

        # Display top-5 predictions
        top5 = outputs.argsort()[-5:][::-1]
        st.subheader("Top 5 Predictions:")
        for idx in top5:
            st.write(f"Class {idx}: {outputs[idx]:.4f}")
