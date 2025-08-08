This document provides detailed instructions for setting up the environment, configuring, and running the `CoolAgent` benchmark.

## Overview

The benchmark is orchestrated by a central shell script (`run_benchmark.sh`) that reads its configuration from a YAML file (`config/default.yaml`). This allows for reproducible, end-to-end execution of benchmark tasks with the `CoolAgent`.

## 1. Prerequisites

Before you begin, ensure you have the following software installed and configured on your system:

*   **Git:** For cloning the repository.
*   **Conda:** For managing the Python environment.
*   **Android SDK:** Required for the `adb` and `emulator` command-line tools.
    *   You must also create an **Android Virtual Device (AVD)** in Android Studio with the name `AndroidWorldAvd`.
*   **`yq`:** A command-line YAML processor used by the runner script.
    *   Installation instructions can be found at: [https://github.com/mikefarah/yq/#install](https://github.com/mikefarah/yq/#install)
*   **Google Gemini API Key:** The `CoolAgent` uses the Gemini family of models. You must have a valid API key.

## 2. Setup and Installation

Follow these steps to set up the project and its dependencies.

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create and Activate Conda Environment:**
    This project uses a Conda environment named `android_world`. Create it using the provided `requirements.txt` file.
    ```bash
    # Create the environment
    conda create -n android_world python=3.10 -y

    # Activate the environment
    conda activate android_world

    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Set Gemini API Key:**
    Export your Gemini API key as an environment variable. The application will not run without it.
    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

## 3. Configuration

All run settings are managed in `config/default.yaml`. To run the `CoolAgent`, ensure the configuration is set correctly.

1.  **Set the Active Agent:**
    In `config/default.yaml`, set `active_agent` to `cool_agent`.

    ```yaml
    agent:
      # Specifies which agent configuration to use from the `configurations` block below.
      active_agent: "cool_agent"
    ```

2.  **Review `CoolAgent` Parameters:**
    Verify the settings within the `agent.configurations.cool_agent` block. The default settings are optimized for robust performance.

    ```yaml
    cool_agent:
      registry_name: "cool_agent"
      model_name: "auto"
      temperature: 0.0
      top_p: 0.95
      max_retry: 3
      # Use -1.0 for dynamic UI stabilization wait (Recommended).
      # Set to a positive float (e.g., 2.0) for a fixed wait time.
      wait_after_action_seconds: -1.0
      enable_safety_checks: true
    ```

## 4. First-Time Emulator Setup

Before running any tasks, you must perform a **one-time setup** to install the required applications and services on the Android emulator.

1.  **Configure for Setup:**
    In `config/default.yaml`, set `perform_emulator_setup` to `"run_once"`.

    ```yaml
    script:
      # ...
      perform_emulator_setup: "run_once"
    ```

2.  **Run the Script:**
    Execute the benchmark script. It will detect that setup is required and perform the necessary installations.
    ```bash
    ./run_benchmark.sh
    ```
    This process will create a hidden file named `.emulator_setup_completed` in the root directory. On subsequent runs, the script will see this file and automatically skip the setup.

3.  **(Optional) Revert Configuration:**
    You can now change `perform_emulator_setup` back to `"skip"` in the config file.

## 5. Running the Benchmark

With the setup and configuration complete, running the `CoolAgent` is straightforward.

1.  **Activate the Environment:**
    Ensure your `android_world` conda environment is active and the `GEMINI_API_KEY` is set.
    ```bash
    conda activate android_world
    export GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

2.  **Execute the Runner Script:**
    ```bash
    ./run_benchmark.sh
    ```
    The script will handle the entire process:
    *   Starting the `AndroidWorldAvd` emulator.
    *   Creating a timestamped run directory for logs and results.
    *   Executing the `CoolAgent` on the configured benchmark suite.

## 6. Understanding the Output

All artifacts from a run are stored in a new, unique directory inside the `runs/` folder. The directory name will look like this: `runs/YYYYMMDD_HHMMSS_cool_agent/`.

Inside this directory, you will find:
*   `benchmark_run.log`: The main log file containing the console output from the entire run.
*   `cool_agent_dump.md`: A static code dump of the agent files, for full reproducibility.
*   **Per-Task Subdirectories:** For each task executed, a separate folder is created (e.g., `ContactsAddContact/`). Inside, you will find:
    *   `screenshots/`: Contains screenshots of the UI at each step (`_before_raw.png`, `_before_marked.png`, `_after_marked.png`).
    *   `llm_interactions.log`: A detailed log of every prompt and response from the Gemini API for that specific task.
    *   `full_analysis.md`: A comprehensive, auto-generated Markdown file that includes the goal, step-by-step reasoning, actions, and summaries, making it easy to debug and analyze a specific task run.

## 7. Troubleshooting

*   **`yq: command not found`:** You have not installed the `yq` utility. Please follow the installation instructions in the Prerequisites section.
*   **`GEMINI_API_KEY environment variable not set`:** You forgot to export your API key before running the script. Run `export GEMINI_API_KEY="..."` and try again.
*   **Emulator Fails to Start:**
    *   Ensure Android Studio and the Android SDK are correctly installed.
    *   Make sure an AVD named `AndroidWorldAvd` exists.
    *   Check the `emulator.log` file (or the path configured in `environment.emulator_log_path`) for detailed error messages.