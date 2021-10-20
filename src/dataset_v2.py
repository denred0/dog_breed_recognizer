import cv2
import matplotlib.pyplot as plt
from torch.utils import data

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pathlib import Path

import numpy as np
import pickle
import os

from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import config


class myDataset(data.Dataset):
    def __init__(self, data, augments=None):
        super().__init__()
        self.imgs, self.labels = data
        self.augments = augments

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = str(self.imgs[idx])
        label = self.labels[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        try:
            if self.augments:
                augmented = self.augments(image=img)
                img = augmented['image']
        except:
            print(img_path)

        return img, label


def get_loaders(source_dir, input_image_size, batch_size, mean, std):
    train_transforms = A.Compose([
        A.Resize(int(input_image_size * 1.2), int(input_image_size * 1.2)),
        A.Rotate(limit=20, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.GaussNoise(p=0.),
        A.OneOf([A.MotionBlur(p=0.5),
                 A.MedianBlur(blur_limit=3, p=0.5),
                 A.Blur(blur_limit=3, p=0.1)], p=0.5),
        A.OneOf([A.CLAHE(clip_limit=2),
                 A.Sharpen(),
                 A.Emboss(),
                 A.RandomBrightnessContrast()], p=0.5),
        A.RandomCrop(input_image_size, input_image_size),
        A.Resize(input_image_size, input_image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], p=1.0)

    val_transforms = A.Compose([
        A.Resize(input_image_size, input_image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ], p=1.0)

    path = Path(source_dir)

    train_val_files = list(path.rglob('*.' + 'jpg'))
    train_val_labels = [path.parent.name for path in train_val_files]

    label_encoder = LabelEncoder()
    label_encoder.fit_transform(train_val_labels)

    # save labels dict to file
    with open('label_encoder.pkl', 'wb') as le_dump_file:
        pickle.dump(label_encoder, le_dump_file)

    # for cl in label_encoder.classes_:
    with open('classes.txt', 'w') as f:
        for item in label_encoder.classes_:
            f.write("%s\n" % item)

    train_files, val_files = train_test_split(train_val_files, test_size=0.15, stratify=train_val_labels)

    train_labels = [path.parent.name for path in train_files]
    train_labels = label_encoder.transform(train_labels)
    train_data = train_files, train_labels

    val_labels = [path.parent.name for path in val_files]
    val_labels = label_encoder.transform(val_labels)
    val_data = val_files, val_labels

    class_weights = []
    count_all_files = 0
    for root, subdir, files in os.walk(source_dir):
        if len(files) > 0:
            class_weights.append(len(files))
            count_all_files += len(files)

    classes_weights = [x / count_all_files for x in class_weights]

    sample_weights = [0] * len(train_files)

    for idx, (data, label) in enumerate(zip(train_files, train_labels)):
        class_weight = classes_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print(classes_weights)

    dataset_train = myDataset(data=train_data, augments=train_transforms)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=4)

    dataset_val = myDataset(data=val_data, augments=val_transforms)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=2)

    return train_loader, val_loader, dataset_train, dataset_val, label_encoder


def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    # mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.229, 0.224, 0.225])
    # std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    # все, что меньше 0 становится = 0, что больше 1 становится равным 1
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


def visuazlize():
    train_loader, val_loader, dataset_train, dataset_val, label_encoder = get_loaders(config.DATA_DIR, config.IMG_SIZE,
                                                                                      config.BATCH_SIZE,
                                                                                      config.MEAN, config.STD)

    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(12, 12), sharey=True, sharex=True)

    for i, fig_x in enumerate(ax.flatten()):
        random_characters = int(np.random.uniform(0, 500))  # i
        im_val, label = dataset_train[random_characters]
        img_label = " ".join(map(lambda x: x.capitalize(), label_encoder.inverse_transform([label])[0].split('_')))
        imshow(im_val.data.cpu(), title=img_label, plt_ax=fig_x)

    plt.show()


if __name__ == '__main__':
    visuazlize()
