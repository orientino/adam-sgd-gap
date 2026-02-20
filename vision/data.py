import os
from io import BytesIO

import numpy as np
import torchvision.transforms as T
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

I1K_TRAIN_SAMPLES = 1_281_167
C10_TRAIN_SAMPLES = 50_000
C5M_SAMPLES = 5_000_000


def get_dataloaders(dataset, **kwargs):
    if dataset == "c10":
        return get_c10_dataloaders(**kwargs)
    elif dataset == "c5m":
        return get_c5m_dataloaders(**kwargs)
    elif dataset == "c5m_imbalanced":
        return get_c5m_imbalanced_dataloaders(**kwargs)
    elif dataset == "i1k_imbalanced":
        return get_i1k_imbalanced_dataloaders(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_c10_dataloaders(
    dir_data,
    batch_size=256,
    n_workers=8,
    aug=True,
):
    transform = [
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    transform_vl = T.Compose(transform[2:])
    transform_tr = T.Compose(transform) if aug else transform_vl
    tr_dataset = CIFAR10(dir_data, train=True, transform=transform_tr, download=True)
    vl_dataset = CIFAR10(dir_data, train=False, transform=transform_vl, download=True)
    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
    )
    vl_loader = DataLoader(
        vl_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
    )
    steps_per_epoch = C10_TRAIN_SAMPLES // batch_size
    return tr_loader, vl_loader, 10, steps_per_epoch


def get_c5m_dataloaders(
    dir_data,
    batch_size=256,
    n_workers=8,
    aug=True,
):
    images, labels = [], []
    for i in range(6):
        data = np.load(os.path.join(dir_data, "cifar-5m", f"part{i}.npz"))
        images.append(data["X"])
        labels.append(data["Y"])
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    indices = np.random.permutation(len(images))[:C5M_SAMPLES]
    images, labels = images[indices], labels[indices]
    transform = [
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    transform_vl = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    transform_tr = T.Compose(transform if aug else [transform[0]] + transform[3:])
    tr_dataset = CIFAR5M(images, labels, transform=transform_tr)
    vl_dataset = CIFAR10(dir_data, train=False, transform=transform_vl, download=True)
    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
    )
    vl_loader = DataLoader(
        vl_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
    )
    steps_per_epoch = C5M_SAMPLES // batch_size
    return tr_loader, vl_loader, 10, steps_per_epoch


def get_c5m_imbalanced_dataloaders(
    dir_data,
    batch_size=256,
    n_workers=8,
    aug=True,
):
    data = np.load(os.path.join(dir_data, "cifar-5m-imbalanced", "c5m_imbalanced.npz"))
    images, labels = data["X"], data["Y"]
    transform = [
        T.ToPILImage(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    transform_vl = T.Compose(
        [
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    transform_tr = T.Compose(transform if aug else [transform[0]] + transform[3:])
    tr_dataset = CIFAR5M(images, labels, transform=transform_tr)
    vl_dataset = CIFAR10(dir_data, train=False, transform=transform_vl, download=True)
    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
    )
    vl_loader = DataLoader(
        vl_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
    )
    steps_per_epoch = len(labels) // batch_size
    return tr_loader, vl_loader, 10, steps_per_epoch


def get_i1k_imbalanced_dataloaders(
    dir_data,
    batch_size=256,
    n_workers=8,
    aug=True,
):
    data = np.load(
        os.path.join(dir_data, "imagenet1k-imbalanced", "i1k_imbalanced.npz"),
        mmap_mode="r",
        allow_pickle=True,
    )
    images, labels = data["X"], data["Y"]
    transform = [
        T.RandomResizedCrop(224, scale=(0.05, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    transform_vl = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    transform_tr = T.Compose(transform) if aug else transform_vl
    tr_dataset = I1KImbalanced(images, labels, transform=transform_tr)
    vl_dataset = (
        wds.WebDataset(
            os.path.join(dir_data, "imagenet1k", "imagenet1k-validation-{00..63}.tar"),
            shardshuffle=False,
            nodesplitter=wds.split_by_node,
        )
        .decode("pil")
        .map(lambda x: (transform_vl(x["jpg"]), int(x["cls"])))
        .with_length(50_000)
    )
    tr_loader = DataLoader(
        tr_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    vl_loader = DataLoader(
        vl_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    steps_per_epoch = len(labels) // batch_size
    return tr_loader, vl_loader, 1000, steps_per_epoch


class CIFAR5M(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


class I1KImbalanced(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(BytesIO(self.images[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_data", type=str, required=True)
    args = parser.parse_args()

    tr_dl, vl_dl, _, _ = get_dataloaders("c10", dir_data=args.dir_data)
    print(f"c10 tr: {len(tr_dl.dataset)}, vl: {len(vl_dl.dataset)}")
    tr_dl, vl_dl, _, _ = get_dataloaders("c5m", dir_data=args.dir_data)
    print(f"c5m tr: {len(tr_dl.dataset)}, vl: {len(vl_dl.dataset)}")
    tr_dl, vl_dl, _, _ = get_dataloaders("c5m_imbalanced", dir_data=args.dir_data)
    print(f"c5m_imbalanced tr: {len(tr_dl.dataset)}, vl: {len(vl_dl.dataset)}")
    tr_dl, vl_dl, _, _ = get_dataloaders("i1k_imbalanced", dir_data=args.dir_data)
    print(f"i1k_imbalanced tr: {len(tr_dl.dataset)}, vl: {len(vl_dl.dataset)}")
