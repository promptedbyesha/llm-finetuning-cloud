from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def evaluate_model(model_path, test_sentences):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    for sentence in test_sentences:
        output = generator(sentence, max_length=50, num_return_sequences=1)
        print(f"Input: {sentence}\nOutput: {output[0]['generated_text']}\n")

if __name__ == "__main__":
    test_samples = [
        "How does fine-tuning work?",
        "Explain parameter-efficient methods like LoRA."
    ]
    evaluate_model("models/llm-finetuned", test_samples)
