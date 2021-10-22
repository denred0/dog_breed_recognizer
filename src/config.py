# URL = "https://tfhub.dev/tensorflow/efficientnet/b3/feature-vector/1"
# URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/feature_vector/2" #!!300
# URL = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b3/classification/2"
# URL = "https://tfhub.dev/google/imagenet/nasnet_large/classification/5"
# URL = "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5" #!!299
# URL = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"  # 299 !!
URL = "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5"

IMG_SIZE = 224  # 224
BATCH_SIZE = 8
NUM_CLASSES = 120
NUM_EPOCHS = 10
DATA_DIR = "data/dogs_dataset_prepared"
MODEL_PATH = "model/mobilenet_v3_small_100_224"
LOAD_MODEL = False

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
