# Models

## How to (develop and) optimize models?

The following diagram shows the steps and the cloud services on which the steps run.

![image](https://github.com/user-attachments/assets/06015468-5de2-453e-8a85-5ad15de0c022)

The main workload is executed on the GPU VM since some of TensorRT's optimizations, such as kernel fusion, are specific to the machine's GPU.

1. SSH into the GPU VM through Google Cloud console or with `gcloud compute ssh`
    ```bash
    gcloud compute ssh gpu-vm-name --project=<project-id> --zone=<gpu-vm-zone> --ssh-flag="-A"
    ```

    The `ssh-flag="-A"` forwards your SSH agent. Assuming you have SSH keys set up with your GitHub account, this flag allows you to clone the repo in the next step with SSH.
2. (Once) Configure your Git username and email in the GPU VM
    ```bash
    git config --global user.name "Your Name"
    git config --global user.email "your_github_account_email@example.com"
    ```
3. (Once) Clone this repo into the GPU-VM.
4. (Once) [Authenticate for using the gcloud CLI](https://cloud.google.com/docs/authentication/gcloud) with your **user credentials**.
5. (Once) Set project with command
    ```bash
    gcloud config set project <PROJECT_ID>
    ```
6. (Once) Set GCP project as the quota billing project with the following command:
    ```bash
    gcloud auth application-default login
    gcloud auth application-default set-quota-project <PROJECT_ID>
    ```
7. Install required dependencies.
    ```
    uv sync --group models
    ```
8. Initialize workspace.
   ```
   uv run src/init.py
   ```
9. Optimize the model with TensorRT.
10. Store the optimized TensorRT model called the TensorRT engine in the Triton model repository following Triton's required directory structure.

> [!IMPORTANT]
> To optimize or run code that uses the GPU, coordinate with the rest of the team to avoid issues since the GPU VM is shared.

