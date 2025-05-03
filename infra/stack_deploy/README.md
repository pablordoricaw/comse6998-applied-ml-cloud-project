# Triton Model Deployment for Vertex AI

This directory contains scripts for creating a custom NVIDIA Triton Server container image and deploying models to Google Cloud's Vertex AI using that container.

## Prerequisites

- Google Cloud CLI installed and configured
- Docker installed and configured
- Python 3.10 or later
- Access to a Google Cloud project with necessary permissions
- Artifact Registry and Vertex AI APIs enabled

## Scripts Overview

### 1. `build_triton_container_image.py`

This script pulls an NVIDIA Triton Server image, tags it, and pushes it to Google Cloud Artifact Registry.

**Dependencies:**
```
docker
pyyaml
yaspin
```

### 2. `deploy_model.py`

This script uploads a model to Vertex AI Model Registry and optionally deploys it to an endpoint.

**Dependencies:**
```
google-cloud-aiplatform
pyyaml
prompt-toolkit
yaspin
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install docker pyyaml yaspin google-cloud-aiplatform prompt-toolkit
```

### 2. Configure gcloud CLI

Ensure you've authenticated with Google Cloud and configured Docker to use gcloud as a credential helper:

```bash
gcloud auth login
gcloud auth configure-docker [LOCATION]-docker.pkg.dev
```

## Usage Guide

### Step 1: Build and Push Triton Container Image

```bash
python build_triton_container_image.py \
  --project-id YOUR_PROJECT_ID \
  --location YOUR_REGION \
  --artifact-registry YOUR_ARTIFACT_REGISTRY \
  --triton-image-uri nvcr.io/nvidia/tritonserver:24.12-py3
```

Alternatively, use a Pulumi config file:

```bash
python build_triton_container_image.py \
  --pulumi-config-file PATH_TO_PULUMI_CONFIG \
  --artifact-registry YOUR_ARTIFACT_REGISTRY
```

### Step 2: Deploy Model to Vertex AI

```bash
python deploy_model.py \
  --project-id YOUR_PROJECT_ID \
  --location YOUR_REGION \
  --machine-type g2-standard-4 \
  --accelerator-type nvidia-l4 \
  --accelerator-count 1 \
  --endpoint-name YOUR_ENDPOINT_NAME \
  --model-display-name YOUR_MODEL_NAME \
  --triton-model-repository gs://YOUR_MODEL_REPOSITORY \
  --triton-image-tag 24.12-py3 \
  --triton-artifact-registry YOUR_ARTIFACT_REGISTRY \
  --deploy
```

Alternatively, use a Pulumi config file:

```bash
python deploy_model.py \
  --pulumi-config-file PATH_TO_PULUMI_CONFIG \
  --endpoint-name YOUR_ENDPOINT_NAME \
  --model-display-name YOUR_MODEL_NAME \
  --triton-model-repository gs://YOUR_MODEL_REPOSITORY \
  --deploy
```

## Parameters Explanation

### For `build_triton_container_image.py`:

- `--project-id`: Your Google Cloud project ID
- `--location`: Google Cloud region (e.g., us-central1)
- `--pulumi-config-file`: Path to Pulumi config file (alternative to project-id/location)
- `--triton-image-uri`: Base Triton image to pull (default: nvcr.io/nvidia/tritonserver:22.01-py3)
- `--artifact-registry`: Name of your Artifact Registry repository

### For `deploy_model.py`:

- `--project-id`: Your Google Cloud project ID
- `--location`: Google Cloud region (e.g., us-central1)
- `--machine-type`: VM machine type (e.g., g2-standard-4)
- `--accelerator-type`: GPU type (e.g., nvidia-l4)
- `--accelerator-count`: Number of GPUs
- `--pulumi-config-file`: Path to Pulumi config file (alternative to above parameters)
- `--endpoint-name`: Name for the Vertex AI endpoint
- `--model-display-name`: Display name for your model
- `--vertex-model-id`: (Optional) ID of existing model to use
- `--vertex-model-version`: (Optional) Version of existing model (default: 1)
- `--triton-model-repository`: GCS path to your model repository
- `--triton-image-tag`: Tag of the Triton image (default: 24.12-py3)
- `--triton-artifact-registry`: Artifact Registry repo name (default: ar-cntrs-repo)
- `--deploy`: Flag to deploy the model after uploading

## Required Replacements

When using these scripts, replace the following placeholders:

- `YOUR_PROJECT_ID`: Your Google Cloud project ID
- `YOUR_REGION`: Region where resources will be created (e.g., us-central1)
- `YOUR_ARTIFACT_REGISTRY`: Name of your Artifact Registry repository
- `YOUR_ENDPOINT_NAME`: Name for your Vertex AI endpoint
- `YOUR_MODEL_NAME`: Display name for your model
- `gs://YOUR_MODEL_REPOSITORY`: GCS path to your Triton model repository

## Pulumi Configuration File Format

If using a Pulumi config file, ensure it has the following structure:

```yaml
config:
  gcp:project: YOUR_PROJECT_ID
  gcp:region: YOUR_REGION
  gpu-vm:machine-type: g2-standard-4
  gpu-vm:gpu-type: nvidia-l4
  gpu-vm:gpu-count: 1
```