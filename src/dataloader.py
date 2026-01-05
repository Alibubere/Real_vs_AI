import torch
from torch.utils.data import DataLoader

def collate_fn(batch):

    image, label = zip(*batch)

    image = torch.stack(image)
    label = torch.stack(label)

    return image, label


def get_dataloader(
    dataset, batch_size: int = 4, shuffle: bool = False, num_workers: int = 0
):

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
