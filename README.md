# COMS 6998E Applied ML in the Cloud Project

**Team:**

- Tyler Chang ([@tylerchang](https://github.com/tylerchang))
- Timothy Chang ([@tyc2118](https://github.com/tyc2118))
- Athitheya Gobinathan ([@athith-g](https://github.com/athith-g))
- Pablo Ordorica Wiener ([@pablordoricaw](https://github.com/pablordoricaw))

**Course:** COMSE 6998 - Applied ML in the Cloud

**Semester:** Spring 2025

**Instructors:**

- I-Hsin Chung
- Seetharami Seelam

**TAs:**

- Abhilash Ganga
- Rishita Yadav

## Requirements

This section outlines what you need to do on your local machine for this project.

### `gcloud` - Google Cloud CLI tool

1. Install the [gcloud](https://cloud.google.com/sdk/docs/install) CLI tool.
2. [Authenticate for using the gcloud CLI](https://cloud.google.com/docs/authentication/gcloud) with your **user credentials**.
5. (Once) Set project with command
    ```bash
    gcloud config set project <PROJECT_ID>
    ```
3. Set GCP project as the quota billing project with the following command:

    ```bash
    gcloud auth application-default login
    gcloud auth application-default set-quota-project <PROJECT_ID>
    ```

### `uv` - Python Dependency & Virtual Environment Manager

> [!TIP]
> After installing `uv`, go through the [HOW_TO_UV](./HOW_TO_UV.md) tutorial to become a 🥷.

1. Install [uv](https://docs.astral.sh/uv/#installation) (and now you can uninstall `conda`, `pyenv`, and all the rest and just use `uv` 🙂)
2. In the root directory of the project, where the `pyproject.toml` file exists, create the Python virtual environment and install the dependencies:

    ```bash
    uv sync --all-groups
    ```

