# Infra & Config

The code in this directory is the project's infrastructure-as-code (IaC) and configuration-as-code (CaC).

- IaC is done with Pulumi IaC through its Python SDK with a Google Cloud Storage bucket for the backend.
- CaC is done with Ansible.

The following diagram shows all the infrastructure used for the project.

![image](https://github.com/user-attachments/assets/2fce2ce6-1e7f-4154-8f83-d90e2dba8fb0)


## Requirements

This section outlines what you need to do on your local machine to be able to manage the IaC and CaC for this project.

### `gcloud` - Google Cloud CLI tool

1. Install the [gcloud](https://cloud.google.com/sdk/docs/install) CLI tool.
2. [Authenticate for using the gcloud CLI](https://cloud.google.com/docs/authentication/gcloud) with your user credentials.
3. Set GCP project as the quota billing project with the following command:

    ```bash
    gcloud auth application-default set-quota-project
    ```

### `uv` - Python Dependency & Virtual Environment Manager

1. Install [uv](https://docs.astral.sh/uv/#installation) (and now you can uninstall `conda`, `pyenv`, and all the rest and just use `uv` 🙂)
2. In the root directory of the project, where the `pyproject.toml` file exists, create the Python virtual environment and install the dependencies to manage infra & configs with:

    ```bash
    uv sync --group infra
    ```

> [!NOTE]
> All the commands below have the prefix `uv run` because they assume you've installed Pulumi IaC and Ansible into your Python virtual environment with uv. Alternatively, activate the venv `source .venv/bin/activate` and you can run the commands without the prefix.
>

## IaC

> [!IMPORTANT]
> For the IaC, we are:
> 
> 1. using a single [Pulumi stack](https://www.pulumi.com/docs/iac/concepts/stacks/) called `main`,
> 2. deploying from local instead of a deployment pipeline.
>
> **‼️These decisions allow us to deploy changes faster, but require letting the rest of the team know when new changes will be deployed and shortly after committing the IaC changes deployed.**

### How to Preview Changes

```bash
uv run pulumi preview
```

### How to Deploy Changes

```bash
uv run pulumi up
```

### IaC Backend

We use a Google Cloud Storage bucket as the backend for the IaC. This is purely to make the project as self-contained as possible. 

We've included a Python module, state_backend.py, that creates the Google Cloud Storage bucket and can manage user-level permissions to allow infra local deployments. This module uses the Google Cloud Storage Python SDK and the authentication configured for `gcloud`.

This module intends to simplify recreating the project, if needed, and avoid "secret" steps. With that said, to recreate the infra for the project, the steps to follow are:

0. Have installed the [requirements](#requirements) and have this repo cloned on your local machine.
1. Create a project in GCP, *manually*.
2. Select the project as the `gcloud` project and quota-billing-project
3. Run the `state_backend.py` Python module.
4. [Set the bucket as the Pulumi backend](https://www.pulumi.com/docs/iac/concepts/state-and-backends/#google-cloud-storage) with the command:
    ```bash
    uv run pulumi login gs://<state-bucket>
    ```
4. Deploy the IaC and CaC with the instructions in this README.

> Why manually create the project in GCP and not as part of the Python module using the SDK? The Python module uses the `gcloud` auth config. If you don't have an existing project set as your quota-billing-project for your API calls, the API calls to create the project in the Python code will error.

## CaC

> [!NOTE]
> The config must only be applied after a new infra deployment or after any new configs have been added.

### Before Applying the Config

For the VM to configure,

1. Start the VM.
2. Get the VM external IP
3. Add the IP to the `inventory.yaml` file.

### How to Run

Execute the following command to apply the configs:

```bash
uv run ansible-playbook playbook.yaml -i inventory.yaml
```

