import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import pickle

from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from os import walk
from pathlib import Path

import config

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
# os.environ["TFHUB_CACHE_DIR"] = "C:/Users/Denis/PycharmProjects/cat_breed_recognizer/model/hub"
os.environ["TFHUB_CACHE_DIR"] = "/home/vid/hdd/projects/PycharmProjects/cat_breed_recognizer/model/hub"

img_height = img_width = config.IMG_SIZE

val_transforms = A.Compose([
    A.Resize(config.IMG_SIZE, config.IMG_SIZE),
    A.Normalize(mean=config.MEAN, std=config.STD),
    ToTensorV2()
], p=1.0)

class_names = []
with open("classes.txt") as file:
    lines = file.readlines()
    class_names = [line.rstrip() for line in lines]

model = keras.models.load_model(config.MODEL_PATH)

output_dir = Path('data/inference/output')
if output_dir.exists() and output_dir.is_dir():
    shutil.rmtree(output_dir)
Path(output_dir).mkdir(parents=True, exist_ok=True)

correct = 0
all = 0

input_dir = "data/inference/input"
for root, dirs, files in os.walk(input_dir):
    for folder in tqdm(dirs):

        p = os.path.join(input_dir, folder) + os.path.sep

        _, _, images_list = next(walk(p))

        for img_name in images_list:

            img = cv2.imread(os.path.join(p, img_name), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            augmented = val_transforms(image=img)
            img_array = augmented['image'].unsqueeze(0)

            x_batch_train = img_array.permute(0, 2, 3, 1)
            x_batch_train = tf.convert_to_tensor(np.array(x_batch_train))

            predictions = model.predict(x_batch_train)
            score = tf.nn.softmax(predictions[0])

            d = class_names[np.argmax(score)]

            folder_name = ""
            if d == folder:
                correct += 1
                folder_name = folder
            else:
                folder_name = folder + "  -->  " + d

            # folder_name = folder + "____" + d
            folder_result = Path(output_dir).joinpath(folder_name)
            if not folder_result.exists():
                Path(folder_result).mkdir(parents=True, exist_ok=True)

            shutil.copy(os.path.join(p, img_name), folder_result)

            all += 1

print(f"Accuracy: {round(correct / all * 100, 2)}%")
