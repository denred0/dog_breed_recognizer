import tensorflow as tf

saved_model_dir = "model/efficientnetb3"
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open("b3_lite.tflite", "wb") as f:
    f.write(tflite_model)
