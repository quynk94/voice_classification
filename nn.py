# coding= UTF-8

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import argparse
import os.path as op


def train(args):
    # Prepare the data
    X = np.load('extracted_data/feat.npy')
    y = np.load('extracted_data/label.npy').ravel()

    class_count = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)

    # Build the Neural Network
    model = Sequential()
    model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_count, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Convert label to onehot
    lb = LabelEncoder().fit(y_train)
    y_train = lb.transform(y_train)
    y_test = lb.transform(y_test)
    y_train = np_utils.to_categorical(y_train, class_count)
    y_test = np_utils.to_categorical(y_test, class_count)

    # Train and test
    model.fit(X_train, y_train, epochs=2000, batch_size=64)
    score, acc = model.evaluate(X_test, y_test, batch_size=32)
    model.save(args.model)
    print('Test score:', score)
    print('Test accuracy:', acc)


def predict(args):
    if op.exists(args.model):
        model = keras.models.load_model(args.model)
        predict_feat_path = 'extracted_data/predict_feat.npy'
        predict_filenames = 'extracted_data/predict_filenames.npy'
        filenames = np.load(predict_filenames)
        X_predict = np.load(predict_feat_path)
        pred = model.predict_classes(X_predict, batch_size=32)
        for pair in list(zip(filenames, pred)):
            print(pair)
    elif input('Model not found. Train network first? (Y/n)').lower() in ['y', 'yes', '']:
        train()
        predict(args)


def real_time_predict(args):
    pass


def main(args):
    if args.train:
        train(args)
    elif args.predict:
        predict(args)
    elif args.real_time_predict:
        real_time_predict(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--train', action='store_true',
                        help='train neural network with extracted features')
    parser.add_argument('-m', '--model',             metavar='path',
                        default='models/nn.h5', help='use this model path on train and predict operations')
    parser.add_argument('-e', '--epochs', metavar='N',
                        default=500, help='epochs to train', type=int)
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict files in ./predict folder')
    parser.add_argument('-P', '--real-time-predict', action='store_true',
                        help='predict sound in real time')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose print')
    parser.add_argument('-s', '--log-speed', action='store_true',
                        help='performance profiling')
    parser.add_argument('-b', '--batch-size', metavar='size',
                        default=64, help='batch size', type=int)
    args = parser.parse_args()
    main(args)
