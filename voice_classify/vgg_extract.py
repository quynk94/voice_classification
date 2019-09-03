#!/usr/bin/env python
# coding= UTF-8

import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import shutil
import librosa.display

def extract_feature(inputPath, outputDir):
    print('extracting: ' + inputPath)
    name = inputPath.split(os.path.sep)[-1].split('.')[0]
    outputPath = os.path.sep.join([outputDir, name])
    plt.interactive(False)
    clip, sample_rate = librosa.load(inputPath, sr=None)
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    fig.savefig(outputPath, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')


def parse_audio_files(parent_dir, output_dir, file_ext='*.ogg'):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    sub_dirs = os.listdir(parent_dir)
    sub_dirs.sort()
    for label, sub_dir in enumerate(sub_dirs):
        sub_output_dir = os.path.sep.join([output_dir, sub_dir])
        os.mkdir(sub_output_dir)
        if os.path.isdir(os.path.join(parent_dir, sub_dir)):
            for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
                try:
                    ext_features = extract_feature(fn, sub_output_dir)
                except Exception as e:
                    print("[Error] extract feature error in %s. %s" % (fn, e))
                    continue


def main():
    # Get features and labels
    parse_audio_files('train_audios', 'train_images')

    # Predict new
    parse_audio_files('test_audios', 'test_images')


if __name__ == '__main__':
    main()
