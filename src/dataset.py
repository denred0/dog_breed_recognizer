import os
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms


def get_loaders(train_dir, val_dir, batch_size, image_size):
    print("Getting loaders")

    train_transforms = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transforms)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2, pin_memory=True)

    class_weights = []
    for root, subdir, files in os.walk(train_dir):
        if len(files) > 0:
            class_weights.append(1 / len(files))

    sample_weights = [0] * len(train_dataset)
    for idx, (data, label) in enumerate(tqdm(train_dataset.imgs)):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, pin_memory=True, sampler=sampler)

    return train_loader, val_loader
