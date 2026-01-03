import os 
import yaml
import logging


def setup_logging():
    logging_dir = "logs"

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    full_path = os.path.join(logging_dir,"Pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format= "%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(full_path),logging.StreamHandler()]

    )
    logging.info("Logging initialize successfully")


def main():

    setup_logging()
    
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    data_dir = config["data_dir"]








if __name__ == "__main__":
    main()