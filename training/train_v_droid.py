"""
V-DROID P³ Training Script

This script implements the Pairwise Process Preference (P³) training methodology
from the V-DROID paper to fine-tune a verifier model.

Usage:
1.  Create a dataset of manual task trajectories in `manual_trajectories.jsonl`.
2.  Run this script on a machine with a suitable GPU (e.g., RTX 4090).
    `python train_v_droid.py`
"""

import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# --- Configuration ---
# Model to fine-tune. The paper uses Llama-3.1-8B, but we'll use a smaller,
# highly capable model that is easy to run. Gemma-2B is a great choice.
BASE_MODEL_ID = "google/gemma-2b-it"
# Path to your collected data (now relative to the script's location)
TRAINING_DATA_PATH = "../manual_trajectories.jsonl"
# Where to save the fine-tuned model adapter (in the project root)
OUTPUT_DIR = "../v_droid_verifier_model"
# Training parameters
EPOCHS = 10
LEARNING_RATE = 1.41e-5
BATCH_SIZE = 1 # Process one step at a time


def create_training_dataset(data_path: str) -> Dataset:
    """
    Loads trajectory data and transforms it into a dataset for the SFTTrainer.
    Each sample in the dataset will be a single, complete prompt for a
    positive or negative candidate action.
    """
    prompts = []
    # This is the template your agent uses. It must match exactly.
    prompt_template = """You are controlling an Android device to complete this task: {goal}

Here is the history of actions taken so far:
{history}

Current screen has these elements:
{ui_elements}

Candidate action: {candidate_action}

Is this action helpful for the task? Answer only "Yes" or "No".

Answer: """

    print(f"Loading and processing data from {data_path}...")
    with open(data_path, 'r') as f:
        for line in f:
            step_data = json.loads(line)

            goal = step_data['goal']
            history = "\n".join(step_data['history']) if step_data['history'] else "No actions taken yet."
            ui_elements = step_data['ui_elements_description']
            correct_action_desc = step_data['correct_action_description']
            
            # --- Create the Positive Sample ---
            # The model should learn to output "Yes" for the correct action.
            positive_prompt = prompt_template.format(
                goal=goal,
                history=history,
                ui_elements=ui_elements,
                candidate_action=correct_action_desc
            )
            # The "text" field is what the SFTTrainer uses as the full prompt + completion
            prompts.append({"text": positive_prompt + " Yes"})

            # --- Create Negative Samples ---
            # The model should learn to output "No" for all incorrect actions.
            for incorrect_action_desc in step_data['incorrect_action_descriptions']:
                negative_prompt = prompt_template.format(
                    goal=goal,
                    history=history,
                    ui_elements=ui_elements,
                    candidate_action=incorrect_action_desc
                )
                prompts.append({"text": negative_prompt + " No"})

    print(f"Created {len(prompts)} preference samples.")
    # Create a Hugging Face Dataset object
    return Dataset.from_list(prompts)


def main():
    """Main training function."""
    print("--- Starting V-DROID Verifier Training ---")

    # 1. Create the dataset
    dataset = create_training_dataset(TRAINING_DATA_PATH)

    # 2. Configure Q-LoRA (for memory-efficient 4-bit training)
    # This is crucial for running on a single GPU like an RTX 4090
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 3. Load the base model and tokenizer
    print(f"Loading base model: {BASE_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", # Automatically place model on GPU
        trust_remote_code=True,
    )
    model.config.use_cache = False # Recommended for training
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set padding token

    # 4. Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    # Prepare model for k-bit training and apply LoRA adapter
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print("Applied LoRA adapter to the model.")

    # 5. Configure Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none", # Can be "tensorboard" or "wandb"
    )

    # 6. Initialize the Trainer
    # SFTTrainer is perfect for this task, as it simply trains the model
    # to generate the text provided in the 'text' field of the dataset.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 7. Start Training
    print("Starting model training...")
    trainer.train()
    print("Training complete!")

    # 8. Save the final model
    print(f"Saving fine-tuned model adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    print("--- V-DROID Verifier Training Finished ---")


if __name__ == "__main__":
    main()