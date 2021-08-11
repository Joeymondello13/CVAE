from .contact_map import ContactMapDataset
from .datasplit import train_valid_split


def load_dataset(input_path):
    dataset = ContactMapDataset(
        path=input_path,
        shape=(1, 28, 28),
        dataset_name='contact_map',
        scalar_dset_names=[],
        values_dset_name=None,
        scalar_requires_grad=False,
        in_memory=True,
    )

    split_pct = 0.8
    train_loader, valid_loader = train_valid_split(
        dataset,
        split_pct,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )
    return train_loader, valid_loader

