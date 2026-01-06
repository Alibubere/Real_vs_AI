import os
import yaml
import logging
import torch
from torch.utils.data import random_split
from src.dataset import Real_vs_AI
from src.features.transform import get_resnet50_transform
from src.dataloader import get_dataloader
from src.models.model import get_model, get_optimizer
from src.models.train import train_loop


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
    num_workers = training_cnfg["num_workers"]
    resume = training_cnfg["resume_from_checkpoint"]
    lr = training_cnfg["lr"]
    weight_decay = training_cnfg["weight_decay"]
    num_epochs = training_cnfg["num_epochs"]

    # checkpoint config
    checkpoint = config["checkpoint"]
    checkpoint_dir = checkpoint["dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest = checkpoint["latest"]
    best = checkpoint["best"]
    latest_path = os.path.join(checkpoint_dir, latest)
    best_path = os.path.join(checkpoint_dir, best)

    transform = get_resnet50_transform()

    dataset = Real_vs_AI(data_dir=data_dir, transform=transform)
    logging.info("Dataset loader successfully")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = get_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )  # shuffe is True for training data

    val_loader = get_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device=device)
    optimizer = get_optimizer(model=model, lr=lr, weight_decay=weight_decay)

    train_loop(
        resume=resume,
        model=model,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        latest_path=latest_path,
        best_path=best_path,
    )


if __name__ == "__main__":
    main()
