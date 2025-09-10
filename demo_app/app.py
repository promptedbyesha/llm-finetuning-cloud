import streamlit as st
from transformers import pipeline

st.title("Fine-tuned LLM Demo")

# Load a model pipeline (replace 'gpt2' with your actual model repo id if needed)
generator = pipeline('text-generation', model='promptsbyesha/llm-finetuned-model')


# Prompt input
prompt = st.text_input("Enter prompt:")

# Slider to select number of responses
num_responses = st.slider("Number of responses", min_value=1, max_value=5, value=1)

# Additional context input (optional)
extra_context = st.text_area("Additional context (optional):", "")

if prompt:
    # Step 1: Display original prompt and additional context
    st.markdown(f"**Step 1: Prompt:** {prompt}")
    if extra_context.strip():
        st.markdown(f"**Step 1b: Context:** {extra_context}")

    # Step 2: Combine prompt and context for model inference
    full_prompt = prompt + " " + extra_context if extra_context.strip() else prompt
    results = generator(full_prompt, max_length=100, num_return_sequences=num_responses)

    # Step 3: Display generated outputs
    for i, result in enumerate(results):
        st.write(f"Output {i+1}: {result['generated_text']}")

    # Step 4: Example post-processing (uppercase conversion as placeholder)
    processed_outputs = [result['generated_text'].upper() for result in results]
    st.markdown("**Step 4: Post-processed Outputs:**")
    for i, output in enumerate(processed_outputs):
        st.write(f"Processed Output {i+1}: {output}")
