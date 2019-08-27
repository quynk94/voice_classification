# Voice Classification using Deep Learning

## Dependencies

- Python 3.6
- numpy
- librosa
- pysoundfile
- matplotlib
- scikit-learn
- tensorflow
- keras

## Dataset

Please record your voice then convert audio file to `.ogg` extension, then put files to `data` folder

Example:

```html
├── 001 - Nam
│  ├── nam_1.ogg
│  ├── nam_2.ogg
│  ├── nam_3.ogg
│  ...
...
└── 002 - Quy
   ├── quy_0.ogg
   ├── quy_1.ogg
   ├── quy_2.ogg
   ...
```

Put files you want to predict to `predict` folder

Example:

```html
├── predict_file_1.ogg
├── predict_file_2.ogg
├── predict_file_3.ogg
...
```

## Feature Extraction

Put audio files (`.ogg`) under `data` directory and run the following command:

`python feat_extract.py`

Features and labels will be generated and saved in `extracted_data` directory.

## Classify with SVM

Make sure you have `scikit-learn` installed and `feat.npy` and `label.npy` under `extracted_data` directory. Run `svm.py` and you could see the result.

## Classify with Multilayer Perceptron

Install `tensorflow` and `keras` at first. Run `nn.py` to train and test the network.

## Classify with Convolutional Neural Network

- Run `cnn.py -t` to train and test a CNN. Optionally set how many epochs to train on.
- Predict files by either:
  - Putting target files under `predict/` directory and running `cnn.py -p`
  - Recording on the fly with `cnn.py -P`
