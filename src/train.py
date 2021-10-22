import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import shutil
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
from dataset_v2 import get_loaders
from tqdm import tqdm
from pathlib import Path

import config

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

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


def evaluate_model(val_dataset, model, loss_fn):
    accuracy_metric = keras.metrics.SparseCategoricalAccuracy()
    loss = 0

    for idx, (data, labels) in enumerate(val_dataset):
        data = data.permute(0, 2, 3, 1)
        data = tf.convert_to_tensor(np.array(data))
        labels = tf.convert_to_tensor(np.array(labels))

        y_pred = model(data, training=False)
        loss = loss_fn(labels, y_pred)
        accuracy_metric.update_state(labels, y_pred)

    accuracy = accuracy_metric.result()
    accuracy_metric.reset_states()

    print(f"Accuracy over validation set is {accuracy}")
    print(f"Loss over validation set is {loss}")


def get_last_exp_number(model_name):
    folders = [x[0] for x in os.walk(os.path.join("logs", model_name))][1:]

    if not folders:
        return 0
    else:
        return max([int(x.split("_")[1]) for x in folders]) + 1


def main():
    # train_loader, val_loader = get_loaders(config.DATA_DIR + "train", config.DATA_DIR + "val", config.BATCH_SIZE, config.IMG_SIZE)
    train_loader, val_loader, _, _, _ = get_loaders(config.DATA_DIR, config.IMG_SIZE, config.BATCH_SIZE, config.MEAN,
                                                    config.STD)

    if config.LOAD_MODEL:
        print("Loading model")
        model = keras.models.load_model(config.MODEL_PATH)
    else:
        print("Building model")
        model = get_model(config.URL, config.IMG_SIZE, config.NUM_CLASSES)

    learning_rate_fn = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=2000,
                                                                   decay_rate=0.95,
                                                                   staircase=True
                                                                   )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate_fn)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1)
    # estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    # callbacks = [rlronp, estop]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    exp_number = get_last_exp_number(config.MODEL)

    log_folder = Path(os.path.join("logs", config.MODEL))
    if not log_folder.exists():
        Path(log_folder).mkdir(parents=True, exist_ok=True)

    Path.joinpath(log_folder, "exp_" + str(exp_number)).mkdir(parents=True, exist_ok=True)

    val_acc_max = 0

    for epoch in range(config.NUM_EPOCHS):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train_losses = []
        dataset_size = 0
        running_loss = 0.0

        bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (x_batch_train, y_batch_train) in bar:
            x_batch_train = x_batch_train.permute(0, 2, 3, 1)
            x_batch_train = tf.convert_to_tensor(np.array(x_batch_train))
            y_batch_train = tf.convert_to_tensor(np.array(y_batch_train))

            batch_size = x_batch_train.shape[0]

            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
                train_losses.append(loss_value)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            running_loss += (float(loss_value) * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer._decayed_lr('float32').numpy())

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print("Training loss: %.4f" % (np.round(np.mean(train_losses), 4),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        val_losses = []
        dataset_size_val = 0
        running_loss_val = 0.0

        bar_val = tqdm(enumerate(val_loader), total=len(val_loader))
        for step, (x_batch_val, y_batch_val) in bar_val:
            x_batch_val = x_batch_val.permute(0, 2, 3, 1)
            x_batch_val = tf.convert_to_tensor(np.array(x_batch_val))
            y_batch_val = tf.convert_to_tensor(np.array(y_batch_val))

            batch_size = x_batch_val.shape[0]

            val_logits = model(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_logits)
            val_losses.append(val_loss_value)

            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)

            running_loss_val += (float(val_loss_value) * batch_size)
            dataset_size_val += batch_size

            epoch_loss_val = running_loss_val / dataset_size_val

            bar_val.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss_val)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation loss: %.4f" % (np.round(np.mean(val_losses), 4),))

        if val_acc > val_acc_max:
            model.save(os.path.join("model", config.MODEL))
            val_acc_max = val_acc

            with open(os.path.join("logs", config.MODEL, "exp_" + str(exp_number),
                                   f"e{str(epoch)}_val_loss_{str(np.round(np.mean(val_losses), 4))}_val_acc_{str(np.round(float(val_acc), 4))}.txt"),
                      "w") as file:
                file.write("")


if __name__ == "__main__":
    main()
