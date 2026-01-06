import os
import logging
import torch
from src.models.model import save_checkpoint, load_checkpoint
from src.models.train_utils import train_one_epoch, val_one_epoch


def train_loop(
    resume: bool,
    model,
    optimizer,
    device: torch.device,
    num_epochs: int,
    train_loader,
    val_loader,
    latest_path: str,
    best_path: str,
):

    start_epoch = 1
    checkpoint_to_load = None
    best_val_loss = 10

    if resume:
        latest_exist = os.path.exists(latest_path)
        best_exist = os.path.exists(best_path)

        if latest_exist and best_exist:

            latest_chpt_data = torch.load(latest_path, map_location=device)
            best_chpt_data = torch.load(best_path, map_location=device)

            latest_loss = latest_chpt_data.get("best_val_loss", float("inf"))
            best_loss = best_chpt_data.get("best_val_loss", float("inf"))

            if latest_loss <= best_loss:
                checkpoint_to_load = latest_path
                best_val_loss = latest_loss

            else:
                checkpoint_to_load = best_path
                best_val_loss = best_loss

        elif latest_exist:
            # Only latest exists, load it
            checkpoint_to_load = latest_path
            # Load metadata to correctly set best_val_loss tracker
            ckpt_data = torch.load(latest_path, map_location=device)
            best_val_loss = ckpt_data.get("best_val_loss", best_val_loss)

        elif best_exist:
            # Only best exists, load it
            checkpoint_to_load = best_path
            # Load metadata to correctly set best_val_loss tracker
            ckpt_data = torch.load(best_path, map_location=device)
            best_val_loss = ckpt_data.get("best_val_loss", best_val_loss)

        if checkpoint_to_load:

            try:
                model, optimizer, start_epoch, loaded_best_val_loss = load_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    path=checkpoint_to_load,
                    device=device,
                )
                if (
                    loaded_best_val_loss is not None
                    and loaded_best_val_loss < best_val_loss
                ):
                    best_val_loss = loaded_best_val_loss
                logging.info(
                    f"Checkpoint loaded from {checkpoint_to_load}. Resuming training from epoch {start_epoch}, best loss tracked: {best_val_loss:.4f}"
                )
            except RuntimeError as e:
                if "state_dict" in str(e):
                    logging.warning(
                        f"Model architecture mismatch. Starting from scratch. Error: {e}"
                    )
                    start_epoch = 1
                    best_val_loss = 10.0
                else:
                    raise

        else:
            logging.info(
                f"Resume enabled but no valid checkpoint found. Starting from scratch."
            )

    else:
        logging.info(f"Resume disabled. Starting from scratch.")

    for epoch in range(start_epoch, num_epochs + 1):

        train_avg_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            dataloader=train_loader,
        )

        val_avg_loss, val_acc = val_one_epoch(
            model=model, dataloader=val_loader, device=device, epoch=epoch
        )

        logging.info(
            f"Epoch: [{epoch}/{num_epochs}] "
            f"Train loss: {train_avg_loss:.4f} | Train acc: {train_acc:.4f} | "
            f"Val loss: {val_avg_loss:.4f} | Val acc: {val_acc:.4f}"
        )

        if val_avg_loss < best_val_loss:
            logging.info(
                f"validation loss improved from {best_val_loss:.4f} to {val_avg_loss:.4f}.saving best model."
            )
            best_val_loss = val_avg_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                path=best_path,
                best_val_loss=best_val_loss,
            )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            path=latest_path,
            best_val_loss=best_val_loss,
        )
