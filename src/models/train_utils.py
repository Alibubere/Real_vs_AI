import torch
import logging


def train_one_epoch(
    model,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
):

    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(dataloader):

        try:
            images = images.to(device)
            labels = labels.to(device).long()

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                logging.info(
                    f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"loss: {loss.item():.4f} "
                )

        except RuntimeError as e:
            logging.error(f"Runtime error encountered at batch {batch_idx}: {e}")
            continue

    return running_loss / len(dataloader)


def val_one_epoch(
    model,
    dataloader,
    epoch: int,
    device: torch.device,
):

    running_loss = 0.0
    correct = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(dataloader):

            try:

                images = images.to(device)
                labels = labels.to(device).long()

                logits = model(images)
                loss = criterion(logits, labels)

                running_loss += loss.item()

                if batch_idx % 10 == 0:
                    logging.info(
                        f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                        f"loss: {loss.item():.4f} "
                    )
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            except RuntimeError as e:
                logging.error(f"Runtime error encountered at batch {batch_idx}: {e}")
                continue

    return running_loss / len(dataloader), correct / total
