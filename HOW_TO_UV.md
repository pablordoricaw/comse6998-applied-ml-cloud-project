# Managing Python Dependencies with `uv`: Grouped Installs and Seamless Execution

uv is a modern Python dependency and environment manager that streamlines workflows traditionally handled by `pip` and `venv`. It offers 

- rapid dependency resolution,
- built-in virtual environment management,
- and advanced features like dependency grouping—all while maintaining compatibility with standard Python packaging formats.

Below is a practical tutorial on using uv to manage dependencies efficiently, group them for selective installs, and run Python code without manually activating virtual environments. Comparisons to `pip` and `venv` are included to clarify the advantages and workflow changes.

## **0. What is `pyproject.toml`?**

Before jumping into commands, need to explain what the `pyproject.toml` file:

`pyproject.toml` is the standardized configuration file for modern Python projects. It serves as a central place to declare project metadata (such as name, version, and author), manage dependencies, specify build requirements, and configure development tools like linters and formatters. Written in the TOML format for readability, `pyproject.toml` replaces older approaches like `setup.py`, `setup.cfg`, and `requirements.txt` by unifying configuration, improving consistency, and enhancing project maintainability.

Key sections include:
- `[build-system]`: Declares the build backend and its requirements.
- `[project]`: Contains metadata and core dependencies.
- `[tool.*]`: Stores configuration for various development tools.

This unified approach is now the standard in the Python ecosystem, supported by tools like pip, Poetry, Hatch, and UV.

For the rest of this tutorial (and this project) we'll only look into the `[project]` section.

## **1. Managing Python Dependencies with uv**

- **Add a Dependency:**

  ```
  uv add requests
  ```

  This command will:
  - Create a virtual environment (if one doesn’t exist)
  - Install the package and its dependencies
  - Update `pyproject.toml` and `uv.lock` for reproducibility

- **Remove a Dependency:**

  ```
  uv remove requests
  ```

  This uninstalls the package and cleans up metadata.

- **Update or Pin Versions:**

  ```
  uv add requests=2.31.0
  uv add 'requests<3.0.0'
  ```

## **2. Grouping Dependencies (Dependency Groups)**

uv supports logical grouping of dependencies, allowing you to install only what's needed for a particular context (e.g., development, production, testing).

- **Add to a Specific Group:**

  ```
  uv add --group lint ruff
  uv add --group test pytest
  ```

## `pyproject.toml` After 1. and 2.:**

  ```
  [project]
  dependencies = [
  "requests>=2.31.0"
  ]

  [dependency-groups]
  dev = [
    "black>=24.3.0"
  ]
  lint = [
    "ruff>=0.4.0"
  ]
  test = [
    "pytest>=8.1.1"
  ]
  ```

- **Install All Groups:**

  ```
  uv sync --all-groups
  ```

- **Install Only Selected Groups:**

  ```
  uv sync --only-group test
  uv sync --group dev --group lint
  uv sync --no-group dev
  ```
  - `--only-group` installs just the specified group(s).
  - `--no-group` excludes the specified group(s).

- **Install Dependencies:**

  ```
  uv sync
  ```

  - This command only installs the dependencies not in a particular group. For our example that would be the `requests` library.

## **3. Running Python Code Without Explicitly Activating the venv**

uv allows you to run Python scripts directly, handling environment activation and dependency installation automatically.

- **Run a Script:**
  ```
  uv run my_module.py
  ```

  This command will:
  - Ensure the appropriate Python version is available
  - Create and activate the virtual environment if needed
  - Install all required dependencies (and groups, if specified)
  - Execute your module—all in one step

- **Selective Group Execution:**
  ```
  uv run --group test -m pytest
  uv run --no-group dev app.py
  ```


## **4. Comparison: uv vs. pip + venv**

| Feature                  | pip + venv                                    | uv                                              |
|--------------------------|-----------------------------------------------|-------------------------------------------------|
| Environment Creation     | `python -m venv venv`                         | Automatic on first use, or `uv venv`            |
| Dependency Install       | `pip install ...` (after activating venv)     | `uv add ...` (no manual activation needed)      |
| Dependency Groups        | Manual via separate requirements files        | Native via `--group` or `--dev` flags           |
| Running Code             | Must activate venv (`source venv/bin/activate`)| `uv run -- python ...` (no activation needed)   |
| Lockfile/Reproducibility | Third-party tools (pip-tools, Poetry)         | Built-in lockfile and resolver                  |
| Speed                    | Standard                                      | 10–100x faster, parallelized                    |

- **Key Differences:**
  - `pip` installs packages into the currently active environment; `uv` manages environments and dependencies together, reducing manual steps[1][3][4].
  - `venv` is only for environment creation; `uv` integrates this with dependency management.
  - Grouping dependencies is native in uv, making it easy to install only what you need for a given workflow[4][6].


