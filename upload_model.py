from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    repo_id="promptsbyesha/llm-finetuned-model",  # Your Hugging Face repo name
    folder_path="models/llm-finetuned",           # Folder with your saved model files
    path_in_repo="",                              # Upload to the root of the repo
    repo_type="model"
)
print("Upload completed")