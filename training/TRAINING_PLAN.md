# V-DROID Verifier Model Training Plan

This document outlines the steps required to train a verifier model for the V-DROID agent, following the Pairwise Process Preference (P³) methodology described in the paper.

## Phase 1: Data Collection

The goal of this phase is to create a dataset of task trajectories. The quality of the trained model is directly dependent on the quality and diversity of this data.

1.  **Modify the Agent for Data Logging:**
    *   Temporarily add `print()` statements to the `step()` method in `android_world/agents/custom/v_droid_agent.py`.
    *   At each step, you need to log the following information to the console:
        *   The current `goal`.
        *   The current `history` list.
        *   The `ui_elements` description string.
        *   A numbered list of all candidate action descriptions.

2.  **Perform Manual Task Execution:**
    *   Run the modified agent.
    *   For 2-3 different tasks (e.g., "Create a contact for Jane Doe", "Set an alarm for 7 AM"), manually execute the task step-by-step.
    *   At each step, look at the numbered list of candidate actions printed to the console and identify the **one correct action**.

3.  **Create the Dataset File (`manual_trajectories.jsonl`):**
    *   In the root of the project, create a file named `manual_trajectories.jsonl`.
    *   For **each step** you took, create a JSON object and add it as a new line to the file.
    *   The JSON object must have the following structure:

    ```json
    {
      "goal": "The task goal string for that step.",
      "history": ["Description of action from step 1", "..."],
      "ui_elements_description": "The full string describing the UI elements for this step.",
      "correct_action_description": "The string description of the one action you identified as correct.",
      "incorrect_action_descriptions": ["A list of all the other action descriptions that were not correct."]
    }
    ```

## Phase 2: Model Training

This phase takes place on a server with a powerful GPU (e.g., RTX 4090).

1.  **Prepare the Server Environment:**
    *   Clone your project repository onto the server.
    *   Ensure the `manual_trajectories.jsonl` file is present in the project root.
    *   Install all necessary dependencies by running: `pip install -r requirements.txt`.

2.  **Run the Training Script:**
    *   Execute the training script from the project's root directory:
        ```bash
        python training/train_v_droid.py
        ```
    *   The script will load your data, process it into preference pairs, and fine-tune the `google/gemma-2b-it` model.
    *   This process will take some time depending on the size of your dataset.

3.  **Retrieve the Trained Model:**
    *   Once training is complete, a new directory named `v_droid_verifier_model` will be created in your project root.
    *   This directory contains the fine-tuned model adapter (the result of the training). Copy this entire directory back to your local machine.

## Phase 3: Inference with the Fine-Tuned Model

1.  **Integrate the Model:**
    *   Once you have the `v_droid_verifier_model` directory on your local machine, let me know.
    *   I will provide the code to modify `VDroidAgent` to load and use this local, fine-tuned model for inference instead of making API calls to a generic model.
