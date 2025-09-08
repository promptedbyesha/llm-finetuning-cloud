# Project Usage

## Setting Up
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare dataset in `data/processed/`
3. Run training: `python src/model/train.py --config configs/train_config.yaml`
4. Launch demo app: `streamlit run demo_app/app.py`

## Project Structure
- `models/`: Fine-tuned checkpoints
- `demo_app/`: Demo frontend interface
- `configs/`: Configs and hyperparameters
- `tests/`: Unit tests
