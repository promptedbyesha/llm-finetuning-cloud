from langchain.llms import HuggingFacePipeline
from langchain.chains import SequentialChain

# Load your fine-tuned HuggingFace model as a pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="your-model-name-or-path",           # e.g. "yourusername/your-finetuned-model"
    task="text-generation"
)

# Define functions for each step (customize as needed)
def step_1(input_text):
    # Example step: preprocess input
    return input_text.strip()

def step_2(processed_input):
    # Example step: pass through LLM
    return llm(processed_input)

def step_3(llm_output):
    # Example step: post-process output
    return llm_output.upper()  # just an example

# Create a sequential chain from your steps
steps = [step_1, step_2, step_3]
chain = SequentialChain(chains=steps)
