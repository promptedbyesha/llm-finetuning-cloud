from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub

def run_workflow(prompt):
    llm = HuggingFaceHub(repo_id="facebook/opt-350m", model_kwargs={"temperature": 0.7})
    template = PromptTemplate(input_variables=["topic"], template="Explain {topic} in simple terms.")
    chain = LLMChain(prompt_template=template, llm=llm)
    response = chain.run(topic=prompt)
    print(response)

if __name__ == "__main__":
    run_workflow("parameter-efficient fine-tuning (LoRA)")
