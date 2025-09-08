import os
import logging

def setup_logging(log_dir="logs", log_file="training.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )
    return logging.getLogger()

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Logger is set up and ready.")
