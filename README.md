---
license: apache-2.0
language: en
library_name: transformers
pipeline_tag: text-generation
tags:
  - llm
  - finetuned
  - langchain
  - cloud-deployment
model-index:
  - name: llm-finetuned-model
    results: []
---

# Finetuning an Open-Source LLM

This project adapts large language models to domain-specific tasks, leveraging parameter-efficient techniques (LoRA/QLoRA), cloud deployment, and workflow orchestration.  
This repository contains code for fine-tuning large language models (LLMs) on custom datasets, handling cloud orchestration, and uploading final models to Hugging Face Hub.

## Objective

This project aims to fine-tune large language models (LLMs) efficiently for domain-specific tasks, enabling easy deployment via cloud orchestration and interactive demo applications. It demonstrates advanced techniques like parameter-efficient fine-tuning (LoRA/QLoRA) and streamlined workflow automation.

## Getting Started

- Clone this repository  
- Install Python dependencies  
- See `demo_app/app.py` to launch the demo  

---

## Project Structure

- `src/model/`: Core model training, evaluation, and upload scripts.  
- `configs/train_config.yaml`: Configuration file for training hyperparameters and paths.  
- `models/llm-finetuned/`: Output directory where trained model checkpoints and tokenizer files are saved.  
- `upload_model.py`: Script to upload saved model files to Hugging Face Hub.  
- `src/eval/`: Evaluation scripts for the trained models.  

---

## Setup Instructions

1. Create and activate your Python environment:

    ```
    conda create -n llm-finetuning python=3.10 -y
    conda activate llm-finetuning
    ```

2. Install required dependencies:

    ```
    pip install -r requirements.txt
    ```

3. Configure `configs/train_config.yaml` according to your data paths and training parameters.

---
## Training

Run model training with:
```bash
python src/model/train.py
```

This will:

- Load your dataset from the configured path.  
- Fine-tune the specified model.  
- Save model checkpoints and tokenizer files in `models/llm-finetuned/`.  

---

## Uploading Model to Hugging Face

After training completes, upload your model files with:
```bash
python upload_model.py
```

Ensure your `upload_model.py` points to the correct local folder and Hugging Face repository:

from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
repo_id="your-hf-username/your-model-repo",
folder_path="models/llm-finetuned",
path_in_repo="",
repo_type="model"
)
print("Upload completed")

---

## Streamlit Demo Application

This project also includes a Streamlit app for easy demonstration and testing of the fine-tuned model.

### Running the Streamlit App

1. Install Streamlit if not installed:

    ```
    pip install streamlit
    ```

2. Run the app:

    ```
    streamlit run src/demo_app/app.py
    ```

3. Open the local URL provided by Streamlit in your browser to interact with the model.

### Configuration

- Update API keys or model paths in the app configuration file if needed.  
- Modify the app code to customize UI or add features.  

---

## Git Workflow for Updates

To commit and push changes safely when collaborating or syncing with remote:
```bash
git add .
git commit -m "Describe your changes"
git pull --rebase
```
Resolve conflicts if any, then:
git push

For stuck rebase issues, clear the rebase state with:
For Git Bash or WSL
rm -rf .git/rebase-merge

Or in PowerShell
Remove-Item -Recurse -Force .git\rebase-merge

---

## Troubleshooting

- If Git complains about `index.lock`, delete the lock file:

    ```
    rm -f .git/index.lock
    ```

    Or in PowerShell:

    ```
    Remove-Item .git\index.lock
    ```

- Always commit or stash changes before pulling:

    ```
    git add .
    git commit -m "Save progress"
    git pull --rebase
    ```

---

## Future Scope

- Expand support for additional LLM architectures and datasets.  
- Integrate advanced evaluation metrics and error analysis tools.  
- Develop fully featured web applications for user-friendly model interaction.  
- Optimize cloud deployment pipelines for scalable inference.  
- Implement autoML capabilities for hyperparameter and architecture tuning.  
- Add multilingual and multimodal fine-tuning workflows.

---
## Contact

For questions or issues, open an issue in this repository or reach out via email: [workwitheesha@gmail.com](mailto:workwitheesha@gmail.com)

---