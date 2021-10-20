import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
from dataset_v2 import get_loaders
from tqdm import tqdm

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

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # rlronp = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1)
    # estop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    # callbacks = [rlronp, estop]

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    for epoch in range(config.NUM_EPOCHS):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train_losses = []
        for step, (x_batch_train, y_batch_train) in enumerate(tqdm(train_loader)):
            x_batch_train = x_batch_train.permute(0, 2, 3, 1)
            x_batch_train = tf.convert_to_tensor(np.array(x_batch_train))
            y_batch_train = tf.convert_to_tensor(np.array(y_batch_train))

            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
                train_losses.append(loss_value)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

            # Log every 200 batches.
            if step % 500 == 0 and step != 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print("Training loss: %.4f" % (np.round(np.mean(train_losses), 4),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        val_losses = []
        for x_batch_val, y_batch_val in val_loader:
            x_batch_val = x_batch_val.permute(0, 2, 3, 1)
            x_batch_val = tf.convert_to_tensor(np.array(x_batch_val))
            y_batch_val = tf.convert_to_tensor(np.array(y_batch_val))

            val_logits = model(x_batch_val, training=False)
            val_loss_value = loss_fn(y_batch_val, val_logits)
            val_losses.append(val_loss_value)

            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)

        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Validation loss: %.4f" % (np.round(np.mean(val_losses), 4),))

        model.save(config.MODEL_PATH)


if __name__ == "__main__":
    main()
