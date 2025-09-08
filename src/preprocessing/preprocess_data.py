from transformers import AutoTokenizer

def preprocess_data(data, model_name="facebook/opt-350m"):
    """
    Tokenizes input text data using chosen tokenizer.
    Args:
        data (list of str): Input sentences.
        model_name (str): Pretrained model tokenizer.
    Returns:
        tokenized dataset (dict): Dictionary of tokenized inputs.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenizer(data, padding=True, truncation=True, return_tensors="pt")
    return tokenized

if __name__ == "__main__":
    sample = ["Fine-tuning an open-source LLM.", "This is a sample sentence."]
    tokenized_sample = preprocess_data(sample)
    print(tokenized_sample)
