from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
from imutils import paths
import os

from pyimagesearch.nn.conv import FCHeadNet
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor

# FIX Cuda
import tensorflow as tf
config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

# Config information =============
TRAIN_DATA = 'train_images/'
MODEL_NAME = 'model.model'

# Constant  ======================
WIDTH = 224
HEIGHT = 224
CHANNEL = 3


def main():
    imagePaths, classNames = load_data()
    model, baseModel = get_freezed_model(classNames)
    trainX, testX, trainY, testY = separate_data(imagePaths, classNames)
    train(model, baseModel, trainX, trainY, testX, testY, classNames)
    save_model(model)


def separate_data(imagePaths, classNames):
    print("[INFO] Separating data.......................")
    iap = ImageToArrayPreprocessor()
    aap = AspectAwarePreprocessor(WIDTH, HEIGHT)
    sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
    (data, labels) = sdl.load(imagePaths, verbose=500)
    data = data.astype("float") / 255.0

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.25, random_state=42)

    # convert the labels from integers to vectors
    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)

    if len(classNames) == 2:
        trainY = np.hstack((trainY, 1 - trainY))
        testY = np.hstack((testY, 1 - testY))

    return [trainX, testX, trainY, testY]


def load_data():
    imagePaths = list(paths.list_images(TRAIN_DATA))
    classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
    classNames = [str(x) for x in np.unique(classNames)]
    print('imagePaths: ', len(imagePaths))
    return [imagePaths, classNames]


def get_freezed_model(classNames):
    baseModel = VGG16(weights="imagenet", include_top=False,
                      input_tensor=Input(shape=(WIDTH, HEIGHT, CHANNEL)))
    headModel = FCHeadNet.build(baseModel, len(classNames), 256)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # Freeze all baseModel layers
    for layer in baseModel.layers:
        layer.trainable = False

    # Compile model
    opt = RMSprop(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    return (model, baseModel)

def train(model, baseModel, trainX, trainY, testX, testY, classNames):
    train_model(model, trainX, trainY, testX, testY)
    evaluation_model(model, testX, testY, classNames)

    # Defreeze all baseModel layers
    for layer in baseModel.layers:
        layer.trainable = True
    # Compile model
    opt = SGD(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    train_model(model, trainX, trainY, testX, testY)
    evaluation_model(model, testX, testY, classNames)


def train_model(model, trainX, trainY, testX, testY):
    print("[INFO] training head...")
    model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=25)


def evaluation_model(model, testX, testY, classNames):
    print("[INFO] evaluating after initialization...")
    predictions = model.predict(testX, batch_size=8)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1), target_names=classNames))


def save_model(model):
    model.save(MODEL_NAME)


if __name__ == "__main__":
    main()
