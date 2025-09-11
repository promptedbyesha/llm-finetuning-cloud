import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
import os
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("configs/train_config.yaml")

    model_name = config["model_name"]
    dataset_path = config["dataset_path"]
    output_dir = config["output_dir"]
    learning_rate = float(config["learning_rate"])  # Convert to float
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Fix for tokenizers without a pad_token (e.g., GPT-2)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Example dataset loading; replace with your data loading pipeline
    dataset = load_dataset("json", data_files={"train": dataset_path})

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding="max_length",
            max_length=512
        )
        # Add labels identical to input_ids for causal LM loss computation
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        # Removed evaluation_strategy to avoid error
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)         # Saves model files like pytorch_model.bin, config.json
    tokenizer.save_pretrained(output_dir)  # Saves tokenizer files like tokenizer_config.json, vocab files
    print(f"Model and tokenizer saved in {output_dir}")
    
        # Debug: Check if folder exists after saving
    import os
    if os.path.exists(output_dir):
        print(f"Output directory exists: {output_dir}")
    else:
        print(f"Output directory DOES NOT exist: {output_dir}")

    # Debug: List files in output_dir
    saved_files = os.listdir(output_dir) if os.path.exists(output_dir) else []
    print(f"Files saved in output directory: {saved_files}")


if __name__ == "__main__":
    main()