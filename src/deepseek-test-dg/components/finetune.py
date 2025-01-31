import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Initialize the accelerator
accelerator = Accelerator()

# Define constants for directories
DATA_DIR = "/workspace/src/deepseek-test-dg/entity/data"
MODEL_DIR = "/workspace/src/deepseek-test-dg/entity/model"
OUTPUT_DIR = "/workspace/src/deepseek-test-dg/entity/outputs"

# Define model and dataset configurations
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B-Instruct"
DATASET_NAME = "HuggingFaceH4/Bespoke-Stratos-17k"

# Define training hyperparameters
TRAINING_ARGS = {
    "learning_rate": 2.0e-5,
    "num_train_epochs": 1,
    "max_seq_length": 4096,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "gradient_checkpointing": True,
    "bf16": True,
    "logging_steps": 5,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "output_dir": OUTPUT_DIR,
}


def load_model_and_tokenizer():
    """Load the model and tokenizer from pretrained weights."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_DIR)
    return model, tokenizer


def load_and_preprocess_dataset(tokenizer):
    """Load and tokenize the dataset."""
    dataset = load_dataset(DATASET_NAME, cache_dir=DATA_DIR)

    def tokenize_function(examples):
        joined_messages = [
            " ".join(message["content"] for message in messages)
            for messages in examples["messages"]
        ]
        tokenized_inputs = tokenizer(
            joined_messages,
            padding="max_length",
            truncation=True,
            max_length=TRAINING_ARGS["max_seq_length"],
        )
        tokenized_inputs["labels"] = tokenized_inputs[
            "input_ids"
        ].copy()  # Ensure labels are present
        return tokenized_inputs

    return dataset.map(tokenize_function, batched=True)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def main():
    """Main function to run training."""
    print("Initializing training script...")

    model, tokenizer = load_model_and_tokenizer()
    tokenized_dataset = load_and_preprocess_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=TRAINING_ARGS["output_dir"],
        learning_rate=TRAINING_ARGS["learning_rate"],
        num_train_epochs=TRAINING_ARGS["num_train_epochs"],
        per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
        per_device_eval_batch_size=TRAINING_ARGS["per_device_eval_batch_size"],
        gradient_accumulation_steps=TRAINING_ARGS["gradient_accumulation_steps"],
        gradient_checkpointing=TRAINING_ARGS["gradient_checkpointing"],
        bf16=TRAINING_ARGS["bf16"],
        logging_steps=TRAINING_ARGS["logging_steps"],
        evaluation_strategy=TRAINING_ARGS["evaluation_strategy"],
        eval_steps=TRAINING_ARGS["eval_steps"],
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("test", None),  # Handle missing test set
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training completed.")


if __name__ == "__main__":
    main()
