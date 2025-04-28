import argparse
from dataclasses import dataclass
from typing import Optional
import yaml

from google.cloud import aiplatform
from yaspin import yaspin


def get_or_create_endpoint(display_name):
    endpoints = aiplatform.Endpoint.list(filter=f"display_name={display_name}")
    if endpoints:
        print(f"Found existing endpoint: {endpoints[0].name}")
        return endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=display_name)
        print(f"Created new endpoint: {endpoint.name}")
        return endpoint

def get_or_upload_model(
    project_id,
    location,
    model_display_name,
    model_repository,
    vertex_model_id,
    vertex_model_version,
    triton_container_image_uri,
):
    if vertex_model_id:
        return aiplatform.Model(
            f"projects/{project_id}/locations/{location}/models/{vertex_model_id}@{vertex_model_version}"
        )
    else:
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_repository,
            version_aliases=["latest"],
            serving_container_image_uri=triton_container_image_uri,
            serving_container_predict_route="/v2/models/model/infer",
            serving_container_health_route="/v2/health/ready",
            serving_container_args=[
                "--model-control-mode=explicit",
                f"--load-model={model_display_name}",
            ],
        )
        return model


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-id", required=False, help="GCP project id where to create resources"
    )
    parser.add_argument(
        "--location", required=False, help="GCP location where to create the resources."
    )
    parser.add_argument(
        "--machine-type", required=False, help="GCP machine type e.g. g2-standard-4"
    )
    parser.add_argument(
        "--accelerator-type",
        required=False,
        help="GCP accelerator type that is compatible withe machine-type e.g. for g2-standard-4, NVIDIA_TESLA_L4",
    )
    parser.add_argument(
        "--pulumi-config-file",
        required=False,
        help="Pulumi config file with project-id, location, machine-type and gpu-type",
    )
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument(
        "--model-display-name",
        required=True,
        help="Name for the model in the Vertex AI model registry",
    )
    parser.add_argument(
        "--vertex-model-id",
        required=False,
        help="ID of existing model in Vertex AI model registry to deploy, e.g. 5739859715316252672",
    )
    parser.add_argument(
        "--vertex-model-version",
        required=False,
        help="Version of an existing model in Vertex AI model registry to deploy, e.g. 1 (default = 1)",
        default="1",
    )
    parser.add_argument(
        "--triton-model-repository",
        required=False,
        help="Path to the model repository following NVIDIA Triton Server model repository structure e.g. gs://gcs-bkt-model-repository",
        default="gs://gcs-bkt-model-repository",
    )
    parser.add_argument(
        "--triton-container-image-uri",
        required=False,
        help="Path to NVIDIA Triton Server container image.",
        default="us-east1-docker.pkg.dev/applied-ml-cloud-project/ar-cntrs-repo/tritonserver:25.03-py3",
    )
    parser.add_argument("--deploy", required=False, action="store_true", default=False)
    return parser


@dataclass
class DeployConfig:
    project_id: str
    region: str
    machine_type: Optional[str] = None
    accelerator_type: Optional[str] = None
    accelerator_count: Optional[int] = None


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
                deploy_config = DeployConfig(
                    project_id=_pulumi_config.get("gcp:project"),
                    region=_pulumi_config.get("gcp:region"),
                    machine_type=_pulumi_config.get("gpu-vm:machine-type"),
                    accelerator_type=_pulumi_config.get("gpu-vm:gpu-type"),
                    accelerator_count=int(_pulumi_config.get("gpu-vm:gpu-count")),
                )
    else:
        if args.project_id and args.location:
            args_dict = vars(args)
            deploy_config = DeployConfig(
                project_id=args_dict["project_id"],
                region=args_dict["location"],
                machine_type=args_dict.get("machine_type"),
                accelerator_type=args_dict.get("accelerator_type"),
                accelerator_count=int(args_dict.get("accelerator_count")),
            )
        else:
            raise Exception(
                "ERROR: Missing either --pulumi-config-file or both --project-id and --location arguments"
            )

    with yaspin(color="cyan") as spinner:
        aiplatform.init(project=deploy_config.project_id, location=deploy_config.region)

        spinner.write("> Getting or creating endpoint...")
        endpoint = get_or_create_endpoint(args.endpoint_name)
        spinner.write(f"> Endpoint {endpoint.display_name}")

        spinner.write(
            f"> Getting or uploading model '{args.model_display_name}' to Vertex AI Model Registry (can take a few min)..."
        )
        model = get_or_upload_model(
            deploy_config.project_id,
            deploy_config.region,
            args.model_display_name,
            args.triton_model_repository,
            args.vertex_model_id,
            args.vertex_model_version,
            args.triton_container_image_uri,
        )
        spinner.write(
            f"> Model {model.display_name} | id: {model.name} | version: {model.version_id}"
        )

        if args.deploy:
            spinner.write(
                f"> Deploying {args.model_display_name} to {args.endpoint_name} (takes a while)..."
            )
            deployed_model = endpoint.deploy(
                model=model,
                deployed_model_display_name=f"{args.model_display_name}-deployment",
                machine_type=deploy_config.machine_type,
                accelerator_type=deploy_config.accelerator_type.upper().replace(
                    "-", "_"
                ),
                accelerator_count=deploy_config.accelerator_count,
                min_replica_count=1,
                max_replica_count=1,
            )
            spinner.write(f"> Deployed model: {deployed_model}")

        spinner.ok("✅ Done")

>>>>>>> f47a21e (feat(infra): allow --args to configure deployment to Vertex AI and pretty spinner in output)

if __name__ == "__main__":
    main()
