import os
import shutil
import pandas as pd

from pathlib import Path

import cv2
from tqdm import tqdm

from imagededup.methods import PHash


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def train_test_split():
    VAL_PART = 0.15
    DIR = "data/dataset/archive/images"

    for root, subdir, files in os.walk(DIR):

        for dir in tqdm(subdir):
            path_save_train = Path("data/train_val/train/" + dir)
            if path_save_train.exists() and path_save_train.is_dir():
                shutil.rmtree(path_save_train)
            Path(path_save_train).mkdir(parents=True, exist_ok=True)

            path_save_val = Path("data/train_val/val/" + dir)
            if path_save_val.exists() and path_save_val.is_dir():
                shutil.rmtree(path_save_val)
            Path(path_save_val).mkdir(parents=True, exist_ok=True)

            _, _, images_list = next(os.walk(os.path.join(root, dir)))

            for i, image in enumerate(images_list):
                if i % int(1 / VAL_PART) == 0 and i != 0:
                    shutil.copy(os.path.join(root, dir, image), path_save_val)
                else:
                    shutil.copy(os.path.join(root, dir, image), path_save_train)


def copy_all_images():
    DIR = "data/archive/images_without_not_cats_not_dubl_and_big"

    for root, subdir, files in os.walk(DIR):
        for dir in subdir:
            _, _, images_list = next(os.walk(os.path.join(root, dir)))

            for image in tqdm(images_list):
                shutil.copy(os.path.join(root, dir, image),
                            "data/all_images")


def delete_not_cats():
    DIR = "data/archive/images"

    not_cats = get_all_files_in_folder(
        Path('/home/vid/hdd/projects/PycharmProjects/Object-Detection-Metrics/data/yolo4_inference/result'), ['*.jpg'])
    not_cats_list = [x.stem for x in not_cats]

    for root, subdir, files in os.walk(DIR):
        for dir in subdir:
            _, _, images_list = next(os.walk(os.path.join(root, dir)))

            for image in tqdm(images_list):
                if image.split('.')[0] in not_cats_list:
                    shutil.move(os.path.join(root, dir, image), "data/not_cats")


def delete_very_small_images():
    DIR = "data/archive/images_without_not_cats_not_dubl"

    small_images = []
    all_cats = get_all_files_in_folder(Path('data/all_images'), ['*.jpg'])
    for img_p in tqdm(all_cats):
        img = cv2.imread(str(img_p))

        h, w = img.shape[:2]

        if h < 150 or w < 150:
            small_images.append(img_p.name)

    print(len(small_images))

    for root, subdir, files in os.walk(DIR):
        for dir in subdir:
            _, _, images_list = next(os.walk(os.path.join(root, dir)))

            for image in tqdm(images_list):
                if image in small_images:
                    shutil.move(os.path.join(root, dir, image), "data/very_small")


def find_dublicates():
    phasher = PHash()
    encodings = phasher.encode_images(image_dir='data/all_images')
    duplicates0 = phasher.find_duplicates(encoding_map=encodings, max_distance_threshold=0)

    dublicates = []
    for key, value in duplicates0.items():
        if len(value) == 0:
            continue

        lrow = []
        lrow.append(key)
        for v in value:
            lrow.append(v)

        dublicates.append(lrow)

    print(dublicates)

    images_to_remove = []
    inds = []
    all_images = get_all_files_in_folder(Path('data/all_images'), ['*.jpg'])

    for image in tqdm(all_images):
        for i, dubl in enumerate(dublicates):
            if image.name in dubl:
                dubl.remove(image.name)
                for dd in dubl:
                    images_to_remove.append(dd)
                inds.append(i)

    inds = sorted(inds)

    # iretare throw dublicates if ind in saved list thеn continue.
    # If not then add all images to remove_list besides one
    for j, dubl2 in enumerate(dublicates):
        if j in inds:
            continue

        for ddd in dubl2[:-1]:
            images_to_remove.append(ddd)

    DIR = "data/archive/images_without_not_cats"

    for root, subdir, files in os.walk(DIR):
        for dir in subdir:
            _, _, images_list = next(os.walk(os.path.join(root, dir)))

            for image in tqdm(images_list):
                if image in images_to_remove:
                    shutil.move(os.path.join(root, dir, image), "data/dublicates")


def prepare_dogs_dataset():
    DIR = "data/dogs_dataset"

    path_save = Path("data/dogs_dataset_prepared")
    if path_save.exists() and path_save.is_dir():
        shutil.rmtree(path_save)
    Path(path_save).mkdir(parents=True, exist_ok=True)

    for root, subdir, files in os.walk(DIR):

        for dir in subdir:

            new_dir = dir.replace("_", "-")

            new_dir = " ".join(new_dir.split("-")[1:]).title()

            Path(path_save).joinpath(new_dir).mkdir(parents=True, exist_ok=True)

            _, _, images_list = next(os.walk(os.path.join(root, dir)))

            for image in tqdm(images_list):
                shutil.copy(os.path.join(root, dir, image), Path(path_save).joinpath(new_dir))


if __name__ == "__main__":
    # train_test_split()
    # copy_all_images()
    # delete_not_cats()
    # find_dublicates()
    # delete_very_small_images()
    prepare_dogs_dataset()
