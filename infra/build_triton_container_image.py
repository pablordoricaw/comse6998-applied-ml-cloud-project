# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "docker",
#     "pyyaml",
#     "yaspin",
# ]
# ///
import argparse
from functools import partial

import docker
import yaml
from yaspin import yaspin


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-id", required=False, help="GCP project id where to create resources"
    )
    parser.add_argument(
        "--location", required=False, help="GCP location where to create the resources."
    )
    parser.add_argument(
        "--pulumi-config-file",
        required=False,
        help="Pulumi config file with project-id, location, machine-type and gpu-type",
    )
    parser.add_argument(
        "--triton-image-uri",
        required=False,
        default="nvcr.io/nvidia/tritonserver:22.01-py3",
    )
    parser.add_argument("--artifact-registry", required=True)
    return parser

def docker_operation_with_spinner(spinner_text, docker_operation, *args, **kwargs):
    with yaspin(text=spinner_text, color="cyan") as spinner:
        try:
            for line in docker_operation(*args, **kwargs):
                if 'id' in line and 'status' in line:
                    layer = line['id']
                    status = line['status']
                    progress = line.get('progress', '')
                    spinner.text = f"{status}: {layer} {progress}"
                elif 'status' in line:
                    spinner.text = line['status']
            spinner.ok("✅ ")
            return True
        except Exception as e:
            spinner.fail("💥 ")
            print(f"Operation failed: {e}")
            return False

pull_image = partial(
    docker_operation_with_spinner,
    docker_operation=lambda docker_client, image: docker_client.api.pull(image, stream=True, decode=True)
)

push_image = partial(
    docker_operation_with_spinner,
    docker_operation=lambda docker_client, repo: docker_client.images.push(repo, stream=True, decode=True)
)


def main():
    parser = create_argparser()
    args = parser.parse_args()

    if args.pulumi_config_file:
        with open(args.pulumi_config_file, "r") as file:
            try:
                pulumi_config = yaml.safe_load(file)
            except yaml.YAMLError as e:
                print(e)
                exit(-1)

            if "config" in pulumi_config.keys():
                _pulumi_config = pulumi_config["config"]

                project_id = _pulumi_config.get("gcp:project")
                location = _pulumi_config.get("gcp:region")
    else:
        if args.project_id and args.location:
            project_id = args.project_id
            location = args.location
        else:
            raise Exception(
                "ERROR: Missing either --pulumi-config-file or both --project-id and --location arguments"
            )

    if input("Have you configured gcloud CLI credential helper?[y/N]:").lower() in (
        "yes",
        "y",
    ):
        print(f"Pulluing {args.triton_image_uri}")
        docker_client = docker.from_env()

        target_repo = f"{location}-docker.pkg.dev/{project_id}/{args.artifact_registry}/tritonserver"

        pull_image(docker_client=docker_client, image=args.triton_image_uri, spinner_text="Downloading NVIDIA Triton container image...")
        image = docker_client.images.get(args.triton_image_uri)
        image.tag(repository=target_repo)
        push_image(docker_client=docker_client, repo=target_repo, spinner_text="Pushing NVIDIA Triton container image...")

    else:
        print(
            "Follow the steps in the following link to setup gcloud CLI credential helper with you user account: https://cloud.google.com/artifact-registry/docs/docker/authentication#gcloud-helper"
        )


if __name__ == "__main__":
    main()

