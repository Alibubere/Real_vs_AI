import torchvision.models as models
import torch
import logging


def get_model(device: torch.device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)

    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(device)
    return model


def get_optimizer(model, lr, weight_decay):
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    return optimizer


def save_checkpoint(model, optimizer, epoch, path):

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
    }

    torch.save(checkpoint, path)

    logging.info(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device="cuda"):

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 1) + 1

    logging.info(f"Checkpoint loaded from {path}, resuming at epoch {start_epoch}")
    return model, optimizer, start_epoch
