import tensorflow as tf
import os
import config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

converter = tf.lite.TFLiteConverter.from_saved_model(config.MODEL_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("b3_lite.tflite", "wb") as f:
    f.write(tflite_model)
