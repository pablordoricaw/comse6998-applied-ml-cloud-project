# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "docker",
#     "yaspin",
# ]
# ///

import argparse
import os
import signal
import sys
import docker
from yaspin import yaspin

def create_argparser():
    parser = argparse.ArgumentParser(description="Run Triton Inference Server container")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--artifact-registry", required=True)
    parser.add_argument("--region", required=True, default="us-east1", help="GCP region where the Artifact Registry repository is located. (default: us-east1)")
    parser.add_argument("--model-repo", required=True, default="gs://gcs-bkt-model-repository", help="(default: gs://gcs-bkt-model-repository")
    parser.add_argument("--model", required=True, default="resnet50_pruned", help="(default resnet50_pruned")
    parser.add_argument("--triton-image-tag", default="24.12-py3", help="(default: 24.12-py3"))
    parser.add_argument("--detach", action="store_true", help="Run container in background")
    return parser

def build_container_config(args):
    image_path = (
        f"{args.region}-docker.pkg.dev/"
        f"{args.project_id}/{args.artifact_registry}/"
        f"tritonserver:{args.triton_image_tag}"
    )
    host_gcloud_path = os.path.expanduser("~/.config/gcloud")
    container_gcloud_path = "/root/.config/gcloud"
    
    return {
        "image": image_path,
        "name": "tritonserver",
        "remove": True,
        "ports": {
            "8000/tcp": 8000,
            "8001/tcp": 8001,
            "8002/tcp": 8002
        },
        "volumes": {
            host_gcloud_path: {
                "bind": container_gcloud_path,
                "mode": "ro"
            }
        },
        "environment": {
            "GOOGLE_APPLICATION_CREDENTIALS": f"{container_gcloud_path}/application_default_credentials.json",
            "AIP_MODE": "True"
        },
        "device_requests": [
            docker.types.DeviceRequest(count=1, capabilities=[['gpu']])
        ],
        "command": [
            "--model-repository", args.model_repo,
            "--model-control-mode", "explicit",
            "--load-model", args.model
        ]
    }

def handle_exit(signum, frame, container):
    print("\nStopping Triton Server...")
    container.stop()
    sys.exit(0)

def run_container(docker_client, config, detach):
    try:
        container = docker_client.containers.run(**config, detach=True)
        
        if detach:
            print(f"🚀 Triton Server running in background (container ID: {container.short_id})")
            print("Use 'docker logs -f tritonserver' to view logs")
            return True
        
        # Register signal handler for foreground mode
        signal.signal(signal.SIGINT, lambda s,f: handle_exit(s,f,container))
        
        print("🔄 Streaming Triton Server logs (Ctrl+C to stop):\n")
        for line in container.logs(stream=True, follow=True):
            print(line.decode(errors="replace").strip())
            
        return True
        
    except docker.errors.ImageNotFound:
        print("❌ Error: Triton container image not found")
    except docker.errors.APIError as e:
        print(f"❌ Docker API error: {e.explanation}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
    return False

def main():
    parser = create_argparser()
    args = parser.parse_args()
    docker_client = docker.from_env()
    config = build_container_config(args)
    
    try:
        old_container = docker_client.containers.get("tritonserver")
        old_container.remove(force=True)
    except docker.errors.NotFound:
        pass
    
    print(f"🔧 Starting Triton Server ({'detached' if args.detach else 'foreground'} mode)")
    print(f"📦 Model repository: {args.model_repo}")
    print(f"🤖 Model to load: {args.model}")
    
    if run_container(docker_client, config, args.detach) and not args.detach:
        print("\n🛑 Triton Server stopped")

if __name__ == "__main__":
    main()

