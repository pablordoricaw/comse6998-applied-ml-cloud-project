import streamlit as st
import time
import json
from google.cloud import aiplatform_v1
from google.api import httpbody_pb2
from PIL import Image
import numpy as np

# Initialize Streamlit interface
st.title("Image Classification with TensorRT Optimization")

# Configuration variables
PROJECT_ID = "applied-ml-cloud-project"
LOCATION = "us-east1"
ENDPOINT_ID = "8278086706083659776"
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"

# Initialize the Vertex AI client
client_options = {"api_endpoint": API_ENDPOINT}
prediction_client = aiplatform_v1.PredictionServiceClient(client_options=client_options)

# Format the endpoint resource name
endpoint_path = prediction_client.endpoint_path(
    project=PROJECT_ID,
    location=LOCATION,
    endpoint=ENDPOINT_ID
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    # Make prediction
    if st.button("Classify Image"):
        start_time = time.time()
        
        # Preprocess image - resize to 224x224 which is standard for ResNet
        resized_image = image.resize((224, 224))

        # # Convert to numpy array
        img_array = np.asarray(resized_image)     

        # Handle channel order (HWC to CHW for PyTorch)
        if len(img_array.shape) == 3:  # For RGB images
            img_array = np.transpose(img_array, (2, 0, 1))  # (224,224,3) → (3,224,224)

        # Add batch dimension
        img_array = np.expand_dims(img_array, 0)  # (1,3,224,224)

        # Normalize to FP32
        img_array = img_array.astype(np.float32) / 255.0  # Critical for ResNet50    
        
        # Format payload with the current input name
        payload = {
            "id": "0",
            "inputs": [ 
                {
                    "name": "input.1",
                    "shape": [1, 3, 224, 224],
                    "datatype": "FP32",
                    "parameters": {},
                    "data": img_array.flatten().tolist()
                }
            ]
        }

        # Optional, write the payload to a file to see the size, uncomment the code below
        # payload_file = "instances.json"
        # with open(payload_file, "w") as f:
        #     json.dump(payload, f)
        # print(f"Payload generated at {payload_file}")
        # quit()
        
        # Create HTTP body for raw prediction
        http_body = httpbody_pb2.HttpBody(
            data=json.dumps(payload).encode("utf-8"),
            content_type="application/json"
        )
        
        # Create raw predict request
        request = aiplatform_v1.RawPredictRequest(
            endpoint=endpoint_path,
            http_body=http_body
        )
        
        # Send the raw prediction request
        response = prediction_client.raw_predict(request=request)
        
        # Parse and display the response
        response_json = json.loads(response.http_body.data)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        st.write("Classification Results:")
        st.json(response_json)
        st.write(f"Inference Time: {inference_time:.4f} seconds")