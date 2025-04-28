from dataclasses import dataclass

import pulumi
import pulumi_gcp as gcp

TEAM_MEMBERS = [
    "po2311@columbia.edu",
    "asg2278@columbia.edu",
    "tyc2118@columbia.edu",
    "tc3407@columbia.edu",
]


@dataclass
class InstanceArgs:
    """
    Data class containing parameters for provisioning a GPU VM instance.

    Attributes:
    machine_type (str): The type of machine to deploy.
    machine_number (str): An identifier for the machine.
    region (str): The GCP region for the instance.
    zone (str): The GCP zone within the region.
    gpu_type (str): The type of GPU to attach.
    gpu_count (int): The number of GPUs to attach.
    """

    machine_type: str
    machine_number: str
    region: str
    zone: str
    gpu_type: str
    gpu_count: int


class Instance(pulumi.ComponentResource):
    """
    Represents a GPU VM instance resource on Google Cloud Platform.
    Provisions a compute instance configured with GPU accelerators using the specified parameters.
    """

    def __init__(self, args: InstanceArgs, opts: pulumi.ResourceOptions = None):
        """
        Initializes the GPU VM instance with the given configuration arguments.

        Parameters:
        args (InstanceArgs): The configuration parameters for the instance.
        opts (ResourceOptions, optional): Additional resource options. Defaults to None.
        """
        machine_type_short = args.machine_type.split("-")[0]
        gpu_type_short = args.gpu_type.split("-")[-1]
        name = f"gce-{machine_type_short}-{gpu_type_short.lower()}-{args.region}-{args.machine_number}"

        self.instance = gcp.compute.Instance(
            resource_name=name,
            name=name,
            zone=f"{args.region}-{args.zone}",
            machine_type=args.machine_type,
            allow_stopping_for_update=True,
            guest_accelerators=[
                gcp.compute.InstanceGuestAcceleratorArgs(
                    count=args.gpu_count, type=args.gpu_type
                )
            ],
            boot_disk=gcp.compute.InstanceBootDiskArgs(
                initialize_params=gcp.compute.InstanceBootDiskInitializeParamsArgs(
                    image="projects/mpi-nvidia-ngc-public/global/images/nvidia-gpu-cloud-vmi-base-2024-10-2-x86-64",
                    enable_confidential_compute=False,
                    size=300,  # deep learning images are like 50GBs...
                    type="pd-balanced",
                )
            ),
            network_interfaces=[
                gcp.compute.InstanceNetworkInterfaceArgs(
                    network="default",
                    access_configs=[
                        gcp.compute.InstanceNetworkInterfaceAccessConfigArgs(
                            network_tier="PREMIUM"
                        )
                    ],
                )
            ],
            scheduling=gcp.compute.InstanceSchedulingArgs(
                automatic_restart=False,
                on_host_maintenance="TERMINATE",
                preemptible=False,
                provisioning_model="STANDARD",
            ),
            opts=opts,
        )


if __name__ == "__main__":
    gcp_config = pulumi.Config("gcp")
    gpu_vm_config = pulumi.Config("gpu-vm")

    config = {
        "gcp": {
            "project": gcp_config.require("project"),
            "region": gcp_config.require("region"),
            "zone": gcp_config.require("zone"),
        },
        "gpu-vm": {
            "machine-number": gpu_vm_config.require("machine-number"),
            "machine-type": gpu_vm_config.require("machine-type"),
            "gpu-type": gpu_vm_config.require("gpu-type"),
            "gpu-count": gpu_vm_config.require("gpu-count"),
        },
    }

    gpu_vm = Instance(
        InstanceArgs(
            machine_number=config["gpu-vm"]["machine-number"],
            machine_type=config["gpu-vm"]["machine-type"],
            gpu_type=config["gpu-vm"]["gpu-type"],
            gpu_count=config["gpu-vm"]["gpu-count"],
            region=config["gcp"]["region"],
            zone=config["gcp"]["zone"],
        ),
    )

    for gcp_user in ["user:" + member for member in TEAM_MEMBERS]:
        user_email = gcp_user.split(":")[-1]
        gpu_vm_iam_member = gcp.compute.InstanceIAMMember(
            f"gpu-vm-iam-ssh-admin-{user_email}",
            project=config["gcp"]["project"],
            instance_name=gpu_vm.instance.id,
            role="roles/compute.instanceAdmin.v1",
            member=gcp_user,
        )

    artifact_repo = gcp.artifactregistry.Repository(
        "ar-cntrs-repo",
        location=config["gcp"]["region"],
        repository_id="ar-cntrs-repo",
        description="Artifact repository for Docker images",
        format="DOCKER",
        docker_config={
            "immutable_tags": False,
        },
    )

    model_repo_bkt = gcp.storage.Bucket(
        "gcs-bkt-model-repository",
        name="gcs-bkt-model-repository",
        location=config["gcp"]["region"],
        storage_class="STANDARD",
        autoclass=gcp.storage.BucketAutoclassArgs(enabled=False),
        versioning=gcp.storage.BucketVersioningArgs(enabled=False),
        force_destroy=True,
        soft_delete_policy=gcp.storage.BucketSoftDeletePolicyArgs(
            # Disable the soft delete policy on the bucket to allow immediately to delete objects.
            # https://www.pulumi.com/registry/packages/gcp/api-docs/storage/bucket/#bucketsoftdeletepolicy
            retention_duration_seconds=0
        ),
        default_event_based_hold=False,
        enable_object_retention=False,
        uniform_bucket_level_access=True,
        hierarchical_namespace=gcp.storage.BucketHierarchicalNamespaceArgs(
            enabled=True
        ),
    )
    # model_repo_bkt_iam_member = gcp.storage.BucketIAMMember(
    #     f"model_repo_bkt_iam_member-{user_email}",
    #     bucket=model_repo_bkt,
    #     role="roles/storage.buckets.get",
    #     member=gcp_user,
    # )
    model_repo_bkt_iam_member = gcp.storage.BucketIAMMember(
        "model_repo_bkt_iam_member",
        bucket=model_repo_bkt,
        role="roles/storage.admin",
        member=f"projectOwner:{config['gcp']['project']}",
    )

    tensorrt_bkt = gcp.storage.Bucket(
        "gcs-bkt-tensorrt",
        name="gcs-bkt-tensorrt",
        location=config["gcp"]["region"],
        storage_class="STANDARD",
        autoclass=gcp.storage.BucketAutoclassArgs(enabled=False),
        versioning=gcp.storage.BucketVersioningArgs(enabled=True),
        force_destroy=True,
        soft_delete_policy=gcp.storage.BucketSoftDeletePolicyArgs(
            # Disable the soft delete policy on the bucket to allow immediately to delete objects.
            # https://www.pulumi.com/registry/packages/gcp/api-docs/storage/bucket/#bucketsoftdeletepolicy
            retention_duration_seconds=0
        ),
        default_event_based_hold=False,
        enable_object_retention=False,
        uniform_bucket_level_access=True,
        hierarchical_namespace=gcp.storage.BucketHierarchicalNamespaceArgs(
            enabled=False
        ),
    )
