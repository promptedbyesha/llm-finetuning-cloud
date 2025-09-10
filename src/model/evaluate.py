from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import yaml

def load_config(config_path):
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config("configs/train_config.yaml")

    model_dir = config["output_dir"]  # Use fine-tuned model output directory
    test_dataset_path = "data/processed/test.json"  # Adjust path to your test data

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    dataset = load_dataset("json", data_files={"test": test_dataset_path})
    
    def tokenize_function(examples):
        tokenized = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./eval_output",
        per_device_eval_batch_size=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")

if __name__ == "__main__":
    main()