import streamlit as st
from transformers import pipeline

st.title("Fine-tuned LLM Demo")

# Load a model pipeline (replace 'gpt2' with your actual model repo id if needed)
generator = pipeline('text-generation', model='gpt2')

prompt = st.text_input("Enter prompt:")

# Add a slider to select number of responses
num_responses = st.slider("Number of responses", min_value=1, max_value=5, value=1)

if prompt:
    results = generator(prompt, max_length=100, num_return_sequences=num_responses)
    for i, result in enumerate(results):
        st.write(f"Output {i+1}: {result['generated_text']}")
