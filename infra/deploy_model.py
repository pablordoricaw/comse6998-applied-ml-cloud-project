from google.cloud import aiplatform
from typing import Dict, Optional, Sequence, Tuple

# Initialization process to connect to Vertex AI
def setup_vertex_ai(project_id, location):
    aiplatform.init(
        project=project_id,
        location=location
    )

# Either get an existing endpoint or create one if it doesn't exist
def get_or_create_endpoint(endpoint_name): 
    endpoints = aiplatform.Endpoint.list(filter=f'display_name={endpoint_name}')
    
    if endpoints:
        print(f"Found existing endpoint: {endpoints[0].name}")
        return endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
        print(f"Created new endpoint: {endpoint.name}")
        return endpoint

# Upload / register model to Vertex AI Model Registry
def upload_model(display_name, artifact_uri, container_image_uri):
    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=container_image_uri,
        serving_container_predict_route="/v2/models/model/infer",
        serving_container_health_route="/v2/health/ready"
    )
    print(f"Uploaded model: {model.name}")
    return model

def deploy_model(
    model,
    endpoint,
    machine_type,
    accelerator_type,
    deployed_model_display_name,
    accelerator_count=1,
    min_replica_count=1,
    max_replica_count=1,
    traffic_percentage=100,
    traffic_split=None,
    metadata=None,
    sync=True
):
        
    deployed_model = endpoint.deploy(
        model=model,
        deployed_model_display_name=deployed_model_display_name,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        traffic_percentage=traffic_percentage,
        traffic_split=traffic_split,
        metadata=metadata,
        sync=sync
    )
    
    if sync:
        print(f"Deployed model to endpoint {endpoint.name}")
    else:
        print(f"Deploying model to endpoint {endpoint.name}")
    
    return deployed_model

def main():
    # Project settings
    PROJECT_ID = "applied-ml-cloud-project"
    LOCATION = "us-east1"
    MACHINE_TYPE="g2-standard-8"
    ACCLERATOR_TYPE="NVIDIA_TESLA_L4"
    
    # Initialize Vertex AI
    setup_vertex_ai(PROJECT_ID, LOCATION)
    
    # Get or create endpoint
    endpoint = get_or_create_endpoint("image-classification-endpoint")
    

    # Upload model - TODO: UPDATE PATHS WHEN MODEL TEAM IS DONE AND STORED
    model = upload_model(
        display_name="tensorrt-optimized-model",
        artifact_uri="gs://gcs-bkt-model-repository/model-directory",
        container_image_uri="us-east1-docker.pkg.dev/applied-ml-cloud-project/ar-cntrs-repo/tritonserver:25.03-trtllm-python"
    )
    
    # Deploy model
    deployment = deploy_model(
        model=model, 
        endpoint=endpoint, 
        machine_type=MACHINE_TYPE, 
        accelerator_type=ACCLERATOR_TYPE, 
        deployed_model_display_name="tensorrt-optimized-deployment"
    )
    
    print(f"Model deployment complete. Endpoint: {endpoint.resource_name}")

if __name__ == "__main__":
    main()
