#!/usr/bin/env python
# coding= UTF-8

import glob
import os
import librosa
import numpy as np
import soundfile as sf
import sounddevice as sd
import queue
import time
from scipy.io.wavfile import write

N_MFCC = 40
N_CHROMA = 12
N_MELS = 128
N_BANDS = 7
N_TONNETZ = 6
N_CENTROID = 1
N_BANDWIDTH = 1
N_FLATNESS = 1
N_ROLLOFF = 1
N_POLY_FEATURES = 3
N_ZERO_CROSSING_RATE = 1
N_RMS = 1
N_OUTPUT = N_MFCC + N_CHROMA * 3 + N_MELS + N_BANDS + N_TONNETZ + N_CENTROID + \
    N_BANDWIDTH + N_FLATNESS + N_ROLLOFF + \
    N_POLY_FEATURES + N_ZERO_CROSSING_RATE + N_RMS


def extract_feature(file_name=None):
    if file_name:
        print('Extracting', file_name)
        X, sample_rate = sf.read(file_name, dtype='float32')
    else:
        device_info = sd.query_devices(None, 'input')
        sample_rate = int(device_info['default_samplerate'])
        q = queue.Queue()
        def callback(i, f, t, s): q.put(i.copy())
        data = []
        print('Start recording')
        start = time.time()
        with sd.InputStream(samplerate=sample_rate, callback=callback):
            while True:
                if len(data) < 150000:
                    data.extend(q.get())
                else:
                    print('Stop recording')
                    break
        # sd.play(data, sample_rate)
        print('Record time: ', time.time() - start)
        X = np.array(data)

    if X.ndim > 1:
        X = X[:, 0]
    X = X.T

    # short term fourier transform
    stft = np.abs(librosa.stft(X))

    # mfcc (mel-frequency cepstrum)
    mfccs = np.mean(librosa.feature.mfcc(
        y=X, sr=sample_rate, n_mfcc=N_MFCC).T, axis=0)

    # chroma
    chroma_stft = np.mean(librosa.feature.chroma_stft(
        S=stft, sr=sample_rate, n_chroma=N_CHROMA).T, axis=0)
    chroma_cqt = np.mean(librosa.feature.chroma_cqt(
        y=X, sr=sample_rate, n_chroma=N_CHROMA).T, axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(
        y=X, sr=sample_rate, n_chroma=N_CHROMA).T, axis=0)

    # melspectrogram
    melspectrogram = np.mean(librosa.feature.melspectrogram(
        X, sr=sample_rate, n_mels=N_MELS).T, axis=0)

    # spectral contrast
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(
        S=stft, sr=sample_rate).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(
        S=stft, sr=sample_rate).T, axis=0)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(
        S=stft, sr=sample_rate).T, axis=0)
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(
        S=stft).T, axis=0)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(
        S=stft, sr=sample_rate).T, axis=0)

    poly_features = np.mean(librosa.feature.poly_features(
        S=stft, sr=sample_rate, order=2).T, axis=0)
    zero_crossing_rate = np.mean(
        librosa.feature.zero_crossing_rate(y=X).T, axis=0)

    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    rms = np.mean(librosa.feature.rms(S=stft).T, axis=0)

    return np.hstack([
        mfccs,
        chroma_stft,
        chroma_cqt,
        chroma_cens,
        melspectrogram,
        spectral_contrast,
        spectral_centroid,
        spectral_bandwidth,
        spectral_flatness,
        spectral_rolloff,
        poly_features,
        zero_crossing_rate,
        tonnetz,
        rms
    ])


def parse_audio_files(parent_dir, file_ext='*.ogg'):
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    features, labels = np.empty((0, N_OUTPUT)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                try:
                    ext_features = extract_feature(fn)
                except Exception as e:
                    print("[Error] extract feature error in %s. %s" % (fn, e))
                    continue
                features = np.vstack([features, ext_features])
                labels = np.append(labels, label)
            print("extract %s features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype=np.int)


def parse_predict_files(parent_dir, file_ext='*.ogg'):
    features = np.empty((0, N_OUTPUT))
    filenames = []
    test_files = glob.glob(os.path.join(parent_dir, file_ext))
    test_files.sort()
    for fn in test_files:
        ext_features = extract_feature(fn)
        features = np.vstack([features, ext_features])
        filenames.append(fn)
        print("extract %s features done" % fn)
    return np.array(features), np.array(filenames)


def main():
    # Get features and labels
    features, labels = parse_audio_files('data')
    np.save('extracted_data/feat.npy', features)
    np.save('extracted_data/label.npy', labels)

    # Predict new
    features, filenames = parse_predict_files('predict')
    np.save('extracted_data/predict_feat.npy', features)
    np.save('extracted_data/predict_filenames.npy', filenames)


if __name__ == '__main__':
    main()
