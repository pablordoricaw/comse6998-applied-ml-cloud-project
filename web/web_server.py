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
import pandas as pd
import threading
import concurrent.futures

# Initialize Vertex AI
aiplatform.init(
    project="applied-ml-cloud-project",
    location="us-east1",  # Default location, will be overridden based on model
)

# Model endpoint configurations
model_configs = {
    "base": {
        "endpoint_id": "3495263901816193024",
        "data_type": "FP32",
        "input_name": "input.1",
        "location": "us-east1", 
        "batch_size": 32 # Base model only takes in batch size of 32, idk why
    },
    "pruned": {
        "endpoint_id": "5975947951543943168",
        "data_type": "FP32",
        "input_name": "input.1",
        "location": "us-central1", 
        "batch_size": 1  
    },
    "sparse": {
        "endpoint_id": "1710114415145123840",
        "data_type": "FP32",
        "input_name": "input.1",
        "location": "us-east4", 
        "batch_size": 1 
    },
    "quantized": {
        "endpoint_id": "8963319944699707392",
        "data_type": "FP32",
        "input_name": "input.1",
        "location": "us-west1",
        "batch_size": 1
    }
}

# Project details
PROJECT_ID = "applied-ml-cloud-project"
PROJECT_NUMBER = "527470660206"

# Get Google credentials and access token
def get_access_token():
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(Request())
    return credentials.token

# Function to preprocess image
def preprocess_image(img, data_type="FP32", batch_size=32):
    # Resize to 224x224
    img = img.resize((224, 224))
    
    # Convert to array
    img_array = np.asarray(img).astype(np.float32)
    
    if data_type == "FP32":
        # Normalize (0-1)
        img_array = img_array / 255.0
    elif data_type == "INT8":
        # Scale to INT8 range (0-255)
        img_array = img_array.astype(np.uint8)
    
    # Convert from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Create batch with appropriate size
    if batch_size > 1:
        img_batch = np.repeat(img_array, batch_size, axis=0)
    else:
        # Keep as single image if batch_size is 1
        img_batch = img_array
    
    return img_batch

# Function to make prediction with a specific model
def predict_with_model(img_batch, model_name, access_token):
    model_config = model_configs[model_name]
    endpoint_id = model_config["endpoint_id"]
    data_type = model_config["data_type"]
    input_name = model_config["input_name"]
    location = model_config["location"]  # Get location specific to this model
    
    # Construct endpoint URL with model-specific location
    endpoint_url = f"https://{endpoint_id}.{location}-{PROJECT_NUMBER}.prediction.vertexai.goog/v1/projects/{PROJECT_ID}/locations/{location}/endpoints/{endpoint_id}:predict"
    
    # Prepare payload
    payload = {
        "inputs": [
            {
                "name": input_name,
                "shape": list(img_batch.shape),
                "datatype": data_type,
                "data": img_batch.flatten().tolist(),
            }
        ]
    }
    
    # Set headers
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    
    # Make prediction
    start_time = time.time()
    response = requests.post(endpoint_url, headers=headers, data=json.dumps(payload))
    end_time = time.time()
    
    # Process response
    results = json.loads(response.text)
    response_time = end_time - start_time

    # To debug if you want to see the full response
    # st.write("Full response structure:")
    # st.json(results)
    
    # Extract predictions
    if 'outputs' in results:
        output_data = results['outputs'][0]['data']
        predictions = np.array(output_data[:1000])  # First 1000 values (first image)
    else:
        predictions = None
    
    return predictions, response_time

# Function to display prediction results
def display_prediction_results(predictions, model_name, response_time):
    st.subheader(f"{model_name.capitalize()} Model Results")
    st.write(f"API Response Time: {response_time:.3f} seconds")
    st.write(f"Model Location: {model_configs[model_name]['location']}")  # Display model location
    st.write(f"Batch Size: {model_configs[model_name]['batch_size']}")  # Display batch size
    
    if predictions is not None:
        # Get top prediction
        top_class_index = np.argmax(predictions)
        confidence = predictions[top_class_index]
        
        st.write(f"Top Prediction: {labels[top_class_index]} (Confidence: {confidence:.4f})")
        
        # Show top 5 predictions
        st.write("Top 5 Predictions:")
        top_indices = np.argsort(predictions)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            st.write(f"{i+1}. {labels[idx]} (Confidence: {predictions[idx]:.4f})")
    else:
        st.error("Error processing predictions")
    
    st.markdown("---")

# Function to run predictions for a specific model
def run_model_prediction(model_name, img_batch, access_token):
    try:
        predictions, response_time = predict_with_model(img_batch, model_name, access_token)
        
        if predictions is not None:
            # Get top prediction
            top_class_index = np.argmax(predictions)
            confidence = predictions[top_class_index]
            
            return {
                "model_name": model_name,
                "predictions": predictions,
                "response_time": response_time,
                "top_class_index": top_class_index,
                "confidence": confidence
            }
        return None
    except Exception as e:
        st.error(f"Error processing {model_name} model: {str(e)}")
        return None

# Streamlit UI
st.title("ResNet Image Classification Comparison")
st.write("Compare different ResNet50 models: Base, Pruned, Sparse, and Quantized")

# Upload image
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=150)
    
    # Model selection
    model_selection = st.radio(
        "Select model(s) to use for classification:",
        ["Base", "Pruned", "Sparse", "Quantized", "Compare All"]
    )
    
    if st.button("Classify"):
        # Get access token
        access_token = get_access_token()
        
        if model_selection == "Compare All":
            # Run all models and compare
            st.subheader("Comparing All Models")
            st.write("Running all models in parallel...")
            
            # Create a table to store comparison results
            comparison_data = []
            
            # Preprocess images for each model once
            preprocessed_images = {}
            for model_name, config in model_configs.items():
                preprocessed_images[model_name] = preprocess_image(img, config["data_type"], config["batch_size"])
            
            # Execute predictions in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(model_configs)) as executor:
                # Start all prediction tasks
                future_to_model = {
                    executor.submit(run_model_prediction, model_name, preprocessed_images[model_name], access_token): model_name 
                    for model_name in model_configs.keys()
                }
                
                # Process results as they complete
                results = []
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        st.error(f"Error with {model_name} model: {str(e)}")
            
            # Display results
            for result in results:
                model_name = result["model_name"]
                predictions = result["predictions"]
                response_time = result["response_time"]
                top_class_index = result["top_class_index"]
                confidence = result["confidence"]
                
                # Add to comparison data
                comparison_data.append({
                    "Model": model_name.capitalize(),
                    "Top Prediction": labels[top_class_index],
                    "Confidence": f"{confidence:.4f}",
                    "Response Time (s)": f"{response_time:.3f}",
                    "Location": model_configs[model_name]["location"],  # Add location to comparison table
                    "Batch Size": model_configs[model_name]["batch_size"]  # Add batch size to comparison table
                })
                
                # Display individual results
                display_prediction_results(predictions, model_name, response_time)
            
            # Display comparison table
            if comparison_data:
                st.subheader("Model Comparison Summary")
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df)
                
                # Create a bar chart for response times
                st.subheader("Response Time Comparison")
                response_times = [float(data["Response Time (s)"]) for data in comparison_data]
                model_names = [data["Model"] for data in comparison_data]
                chart_data = pd.DataFrame({
                    "Model": model_names,
                    "Response Time (s)": response_times
                })
                st.bar_chart(chart_data.set_index("Model"))
        else:
            # Process with selected model
            model_name = model_selection.lower()
            
            # Preprocess image based on model's data type and batch size
            img_batch = preprocess_image(img, model_configs[model_name]["data_type"], model_configs[model_name]["batch_size"])
            
            # Make prediction
            predictions, response_time = predict_with_model(img_batch, model_name, access_token)
            
            # Display results
            display_prediction_results(predictions, model_name, response_time)