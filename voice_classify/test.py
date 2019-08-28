#!/usr/bin/env python
# coding= UTF-8

import feat_extract
import time
import argparse
import numpy as np
import keras
import os.path as op
import sys
import pickle
import config

def predict(args):
    target_names = config.target_names()
    if op.exists(args.model):
        model = load_model(args.model)
        predict_feat_path = 'extracted_data/predict_feat.npy'
        predict_filenames = 'extracted_data/predict_filenames.npy'
        filenames = np.load(predict_filenames)
        X_predict = np.load(predict_feat_path)
        pred = model.predict(X_predict)
        for pair in list(zip(filenames, target_names[pred])):
            print(pair)


def real_time_predict(args):
    target_names = config.target_names()
    if op.exists(args.model):
        model = load_model(args.model)
        while True:
            try:
                features = np.empty((0, 193))
                start = time.time()
                ext_features = feat_extract.extract_feature()
                features = np.vstack([features, ext_features])
                pred = model.predict(features)
                for p in pred:
                    print(target_names[p])
                    if args.verbose:
                        print(
                            'Time elapsed in real time feature extraction: ', time.time() - start)
                    sys.stdout.flush()
            except KeyboardInterrupt:
                parser.exit(0)
            except Exception as e:
                parser.exit(type(e).__name__ + ': ' + str(e))


def load_model(location):
    f = open(location, "rb")
    model = pickle.load(f)
    f.close()
    return model


def main(args):
    if args.predict:
        predict(args)
    else:
        real_time_predict(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-m', '--model', metavar='path',
                        default='models/logistic.pkl', help='use this model path on predict operations')
    parser.add_argument('-p', '--predict', action='store_true',
                        help='predict files in ./predict folder')
    parser.add_argument('-P', '--real-time-predict', action='store_true',
                        help='predict sound in real time')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='verbose print')
    parser.add_argument('-s', '--log-speed', action='store_true',
                        help='performance profiling')
    args = parser.parse_args()
    main(args)
