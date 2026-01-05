import os
import yaml
import logging
from torchvision import transforms
from torch.utils.data import random_split
from src.dataset import Real_vs_AI
from src.dataloader import get_dataloader


def setup_logging():
    logging_dir = "logs"

    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    full_path = os.path.join(logging_dir, "Pipeline.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(full_path), logging.StreamHandler()],
    )
    logging.info("Logging initialize successfully")


def main():

    setup_logging()

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    # data dir config
    data_dir = config["data_dir"]

    # training config
    training_cnfg = config["training"]
    batch_size = training_cnfg["batch_size"]
    num_workers = training_cnfg["num_wokers"]

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = Real_vs_AI(data_dir=data_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = get_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )  # shuffe is True for training data

    val_loader = get_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


if __name__ == "__main__":
    main()
