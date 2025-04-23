# Infra & Config

The code in this directory is the infrastructure-as-code (IaC) and configuration-as-code (CaC) for the project.

- IaC is done with Pulumi IaC through its Python SDK with a Google Cloud Storage bucket for the backend.
- CaC is done with Ansible.

> [!NOTE]
> All the commands below have the prefix `uv run` because they assume you've installed Pulumi IaC and Ansible into your Python virtual environment with uv. Alternatively, activate the venv `source .venv/bin/activate` and you can run the commands without the prefix.
> 

## IaC


### How to Preview Changes

```bash
uv run pulumi preview
```

## CaC

> [!NOTE]
> The config only needs to be applied after a new infra deployment or any new configs added.

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

