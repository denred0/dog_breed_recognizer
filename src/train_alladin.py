import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
# from dataset import get_loaders
from dataset_v2 import get_loaders
from tqdm import tqdm

import config

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# URL = "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1"
# IMG_SIZE = 300  # 224
# BATCH_SIZE = 8
# NUM_CLASSES = 18
# NUM_EPOCHS = 30
# DATA_DIR = "data/data_simpsons" # "data/archive/images"
# MODEL_PATH = "model/efficientnetb3/"
# LOAD_MODEL = False

# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
# os.environ["TFHUB_CACHE_DIR"] = "C:/Users/Denis/PycharmProjects/cat_breed_recognizer/model/hub"
os.environ["TFHUB_CACHE_DIR"] = "/home/vid/hdd/projects/PycharmProjects/cat_breed_recognizer/model/hub"


def get_model(url, img_size, num_classes):
    model = tf.keras.Sequential([
        hub.KerasLayer(url, trainable=True),
        layers.Dense(1000, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.build([None, img_size, img_size, 3])
    model.summary()

    return model


@tf.function
def train_step(data, labels, acc_metric, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    acc_metric.update_state(labels, predictions)


def evaluate_model(val_dataset, model):
    accuracy_metric = keras.metrics.SparseCategoricalAccuracy()
    for idx, (data, labels) in enumerate(val_dataset):
        data = data.permute(0, 2, 3, 1)
        data = tf.convert_to_tensor(np.array(data))
        labels = tf.convert_to_tensor(np.array(labels))
        y_pred = model(data, training=False)
        accuracy_metric.update_state(labels, y_pred)

    accuracy = accuracy_metric.result()
    accuracy_metric.reset_states()
    print(f"Accuracy over validation set is {accuracy}")


def main():
    train_loader, val_loader, _, _, _ = get_loaders(config.DATA_DIR, config.IMG_SIZE, config.BATCH_SIZE, config.MEAN,
                                                    config.STD)
    # train_loader, val_loader = get_loaders(DATA_DIR + "train", DATA_DIR + "val", BATCH_SIZE, IMG_SIZE)

    if config.LOAD_MODEL:
        print("Loading model")
        model = keras.models.load_model(config.MODEL_PATH)
    else:
        print("Building model")
        model = get_model(config.URL, config.IMG_SIZE, config.NUM_CLASSES)

    optimizer = keras.optimizers.Adam(learning_rate=3e-4)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # model.compile(optimizer=optimizer, loss=loss_fn, metrics=train_acc_metric)

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        for idx, (data, labels) in enumerate(tqdm(train_loader)):
            data = data.permute(0, 2, 3, 1)
            data = tf.convert_to_tensor(np.array(data))
            labels = tf.convert_to_tensor(np.array(labels))

            train_step(data, labels, train_acc_metric, model, loss_fn, optimizer)

            if idx % 500 == 0 and idx != 0:
                train_acc = train_acc_metric.result()
                print(f"Accuracy over epoch (so far) {train_acc}")

        evaluate_model(val_loader, model)
        model.save(config.MODEL_PATH)
        train_acc_metric.reset_states()


if __name__ == "__main__":
    main()
