import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
import os
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("configs/train_config.yaml")

    model_name = config["model_name"]
    dataset_path = config["dataset_path"]
    output_dir = config["output_dir"]
    learning_rate = float(config["learning_rate"])
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        # Add labels for loss calculation
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
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

    # Save fine-tuned model and tokenizer to a dedicated models folder
    save_path = "./models/my_finetuned_llm"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()
