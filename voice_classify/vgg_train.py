import argparse
import cv2
import h5py
import numpy as np
import os
import pickle
import random
import time
from imutils import paths
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


WIDTH = 224
HEIGHT = 224
CHANNEL = 3


def main():
    t = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="dataset",
                    help="path to input dataset")
    ap.add_argument("-m", "--model", default="model.pkl",
                    help="path to output model")
    args = vars(ap.parse_args())

    print("[INFO] extracting features...")
    # db = h5py.File("features.hdf5", "r")
    # features = db["features"][:]
    # labels = db["labels"][:]
    # classNames = [str(i) for i in np.unique(labels)]
    features, labels, classNames = extract_features('train_images')

    print("[INFO] spliting train and validation...")
    trainX, trainY, valX, valY = split_train_val(features, labels, 0.75)

    print("[INFO] training...")
    model = train(trainX, trainY)

    print("[INFO] evaluating...")
    evaluate(model, valX, valY, classNames)

    print("[INFO] saving model...")
    save_model(model.best_estimator_, args["model"])

    print("[INFO] done in {} seconds".format(round(time.time() - t, 2)))


def extract_features(path_to_dataset):
    baseModel = VGG16(weights="imagenet",
                      include_top=False,
                      input_shape=(WIDTH, HEIGHT, CHANNEL))
    featuresLength = np.prod(baseModel.layers[-1].output.shape[1:])

    imagePaths = list(paths.list_images(path_to_dataset))
    classNames = np.unique([pt.split(os.path.sep)[-2] for pt in imagePaths])
    random.shuffle(imagePaths)

    bs = 5
    labels = [p.split(os.path.sep)[-2] for p in imagePaths]
    features = []

    for i in np.arange(0, len(imagePaths), bs):
        batchPaths = imagePaths[i:i + bs]
        batchImages = []

        for (_, imagePath) in enumerate(batchPaths):
            image = load_img(imagePath, target_size=(WIDTH, HEIGHT))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)
            batchImages.append(image)

        batchImages = np.vstack(batchImages)
        batchFeatures = baseModel.predict(batchImages, batch_size=bs)
        features.append(batchFeatures.reshape(len(batchImages), featuresLength))

    labels = np.vstack(labels)
    labels = LabelEncoder().fit_transform(labels)
    features = np.vstack(features)

    return features, labels, classNames


def split_train_val(features, labels, train_ratio):
    tmp = np.hstack((features, labels.reshape(len(labels), 1)))
    np.random.shuffle(tmp)
    features = tmp[:, 0:-1]
    labels = tmp[:, -1:].flatten()

    i = int(labels.shape[0] * train_ratio)
    trainX = features[:i]
    trainY = labels[:i]
    valX = features[i:]
    valY = labels[i:]

    return trainX, trainY, valX, valY


def train(trainX, trainY):
    params = {"C": [0.1, 1.0, 10.0, 100.0]}
    model = GridSearchCV(LogisticRegression(solver="lbfgs",
                                            multi_class="auto", max_iter=100), params, cv=3, n_jobs=-1)
    model.fit(trainX, trainY)
    print("[INFO] best hyperparameters: {}".format(model.best_params_))
    return model


def evaluate(model, testX, testY, classNames):
    preds = model.predict(testX)
    print(classification_report(testY, preds, target_names=classNames))


def save_model(model, location):
    f = open(location, "wb")
    pickle.dump(model, f)
    f.close()


if __name__ == '__main__':
    main()
