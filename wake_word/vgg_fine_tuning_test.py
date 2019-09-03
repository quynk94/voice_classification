import argparse
import cv2
import numpy as np
import os
import pickle
from imutils import paths
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from config import target_names
import librosa
import numpy as np
import matplotlib.pyplot as plt
import shutil
import librosa.display
import time
import soundfile as sf
import sounddevice as sd
import queue
import time
import vgg_extract
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import AspectAwarePreprocessor

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

WIDTH = 224
HEIGHT = 224
CHANNEL = 3

iap = ImageToArrayPreprocessor()
aap = AspectAwarePreprocessor(WIDTH, HEIGHT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--images", help="path to input images")
    ap.add_argument("-a", "--audios", help="path to input audios")
    ap.add_argument("-m", "--model", default="model.model",
                    help="path to input model")
    ap.add_argument("-c", "--camera", type=int, default=0)
    # ap.add_argument("-o", "--output",  help="path to output images")
    args = vars(ap.parse_args())

    print("[INFO] loading pre-trained network...")
    model = load_model(args["model"])

    baseModel = VGG16(weights="imagenet",
                      include_top=False,
                      input_shape=(WIDTH, HEIGHT, CHANNEL))

    if args['images'] is not None:
        detect_images(model, baseModel, args["images"])
    elif args['audios'] is not None:
        detect_audios(model, baseModel, args["audios"])
    else:
        detect_real_time(model, baseModel)


def record_input(file_name=None):
    device_info = sd.query_devices(None, 'input')
    sample_rate = int(device_info['default_samplerate'])
    q = queue.Queue()
    def callback(i, f, t, s): q.put(i.copy())
    data = []
    print('Start recording')
    start = time.time()

    with sd.InputStream(samplerate=sample_rate, callback=callback, channels=2):
        while True:
            if len(data) < 150000:
                data.extend(q.get())
            else:
                print('Stop recording')
                break
    sd.play(data, sample_rate)
    print('Record time: ', time.time() - start)
    X = np.array(data)
    if X.ndim > 1:
        X = X[:, 0]
    X = X.T
    clip = np.array(X)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    fig.savefig('test_sample', dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')


def detect_images(model, baseModel, imagesPath):

    imagePaths = list(paths.list_images(imagesPath))
    classNames = target_names()

    labels = [p.split(os.path.sep)[-2] for p in imagePaths]
    preds = []

    for i in np.arange(0, len(imagePaths)):
        imagePath = imagePaths[i]

        image = cv2.imread(imagePath)
        image = aap.preprocess(image)
        image = iap.preprocess(image)
        image = np.expand_dims(image, axis=0) / 255.0

        result = model.predict(image)
        pred = result.argmax(axis=1)[0]

        preds.append(pred)

    labels = np.vstack(labels)
    lb = LabelEncoder().fit(np.vstack(classNames))
    labels = lb.transform(labels)

    print(classification_report(labels,
                                preds, target_names=classNames))


def detect_audios(model, baseModel, audiosPath):
    vgg_extract.parse_audio_files(audiosPath, 'tmpImages')
    imagesPath = 'tmpImages'

    imagePaths = list(paths.list_images('tmpImages'))
    classNames = target_names()

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = aap.preprocess(image)
        image = iap.preprocess(image)
        image = np.expand_dims(image, axis=0) / 255.0

        result = model.predict(image)
        pred = result.argmax(axis=1)[0]
        print(imagePath + ": " + str(classNames[int(pred)]))
    shutil.rmtree('tmpImages')


def detect_real_time(model, baseModel):

    imagePaths = ['test_sample.png']
    classNames = target_names()

    while True:
        record_input()

        imagePath = imagePaths[0]

        image = cv2.imread(imagePath)
        image = aap.preprocess(image)
        image = iap.preprocess(image)
        image = np.expand_dims(image, axis=0) / 255.0

        result = model.predict(image)
        pred = result.argmax(axis=1)[0]

        print(result)
        prob = result[0][pred]
        print(str(classNames[pred]) + " - " + str(prob))


if __name__ == '__main__':
    main()
