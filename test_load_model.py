from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/llm-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

print("Model and tokenizer loaded successfully!")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))