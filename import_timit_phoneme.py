#!/usr/bin/env python

'''
    NAME    : LDC TIMIT Dataset
    URL     : https://catalog.ldc.upenn.edu/ldc93s1
    HOURS   : 5
    TYPE    : Read - English
    AUTHORS : Garofolo, John, et al.
    TYPE    : LDC Membership
    LICENCE : LDC User Agreement
'''




import errno
import os
from os import path
import sys
import tarfile
import fnmatch
import pandas as pd
import subprocess
import numpy as np
from utils import featurize_mfcc, featurize_spectogram

def parse_timit_line(line):
    start_frame, end_frame, label = line.split(' ')
    return int(start_frame), int(end_frame), label.strip('\n')

def check_phoneme(labels, step, labels_per_step, labels_per_winlen):
    init_pos = step * labels_per_step
    end_pos = int(init_pos + (labels_per_winlen))
    init_pos = int(init_pos)
    phoneme = labels[init_pos]
    for i in range(init_pos, end_pos):
        if phoneme != labels[i]:
            phoneme = -1
            break
    # print("check_phoneme: {}, {}, phoneme: {}".format(init_pos, end_pos, phoneme))
    return phoneme

def _preprocess_data(args):

    # Assume data is downloaded from LDC - https://catalog.ldc.upenn.edu/ldc93s1

    datapath = args
    target = path.join(datapath, "TIMIT")
    print("Checking to see if data has already been extracted in given argument: %s", target)

    if not path.isdir(target):
        print("Could not find extracted data", datapath)
        return
    else:
        # is path therefore continue
        print("Found extracted data in: ", target)

    print("Building CSVs")

    # Lists to build CSV files
    train_list_wavs, train_list_phonemes, train_list_texts = [], [], []
    test_list_wavs, test_list_phonemes, test_list_texts = [], [], []

    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*.WAV"):
            full_wav = os.path.join(root, filename)
            print(full_wav)
            mfcc_file_pattern = full_wav[:-4] + ".mfcc"
            full_rif_wav = full_wav[:-4] + "_rif.wav"
            phn_file = full_wav[:-4] + ".PHN"
            txt_file = full_wav[:-4] + ".TXT"
            wav_filesize = path.getsize(full_wav)
            # MFCC
            winlen = 0.025
            winstep = 0.01
            mfcc_features = featurize_mfcc(full_rif_wav, mfcc_dim=26, winlen=winlen, winstep=winstep)
            input_length = mfcc_features.shape[0]
            # Spectogram
            # mfcc_features = featurize_spectogram(full_rif_wav)
            # input_length = mfcc_features.shape[0]
            # Prepare texts
            text = ""
            with open(txt_file, 'r') as f:
                line = f.readline()  # Read only one line
                columns = line.split(' ')
                text = ' '.join(columns[2:])
                text = text[:-1]
                max_length = int(columns[1])
            # Prepare labels
            labels = []
            with open(phn_file, 'r') as f:
                for line in f.readlines():
                    start_frame, end_frame, label = parse_timit_line(line)
                    phn_frames = int(end_frame - start_frame)
                    labels.extend([label] * phn_frames)
                labels_length = len(labels)
            # Create mfcc files
            mfcc_file_index = 0
            step_relation = winlen / winstep
            labels_per_step = labels_length / input_length
            labels_per_winlen =  labels_per_step * step_relation
            print("Step relation: {}, per_step: {}, per_winlen {}, labels_length {}, input_length: {}".format(step_relation, labels_per_step, labels_per_winlen, labels_length, input_length))
            step_max_length = int(input_length - 3)
            for step in range(0, step_max_length):
                mfcc_block = mfcc_features[step]
                # Each block must have the same phoneme
                phn = check_phoneme(labels, step, labels_per_step, labels_per_winlen)
                if phn == 'h#':
                    continue
                if phn != -1:
                    mfcc_file_index += 1
                    mfcc_file = "{}.{}.npy".format(mfcc_file_pattern, mfcc_file_index)
                    np.save(mfcc_file, mfcc_block)
                    if 'train' in full_wav.lower():
                        train_list_wavs.append(mfcc_file)
                        train_list_phonemes.append(phn)
                        train_list_texts.append(text)
                    elif 'test' in full_wav.lower():
                        test_list_wavs.append(mfcc_file)
                        test_list_phonemes.append(phn)
                        test_list_texts.append(text)
                    else:
                        raise IOError
    a = {'wav_filename': train_list_wavs,
         'text': train_list_texts,
         'transcript': train_list_phonemes
         }

    c = {'wav_filename': test_list_wavs,
         'text': test_list_texts,
         'transcript': test_list_phonemes
         }

    all = {'wav_filename': train_list_wavs + test_list_wavs,
          'text': train_list_texts + test_list_texts,
          'transcript': train_list_phonemes + test_list_phonemes
          }

    df_all = pd.DataFrame(all, columns=['text', 'wav_filename', 'transcript'], dtype=int)
    df_train = pd.DataFrame(a, columns=['text', 'wav_filename', 'transcript'], dtype=int)
    df_test = pd.DataFrame(c, columns=['text', 'wav_filename', 'transcript'], dtype=int)

    df_all.to_csv(target+"/timit_phoneme_all.csv", sep=',', header=True, index=False, encoding='ascii')
    df_train.to_csv(target+"/timit_phoneme_train.csv", sep=',', header=True, index=False, encoding='ascii')
    df_test.to_csv(target+"/timit_phoneme_test.csv", sep=',', header=True, index=False, encoding='ascii')

if __name__ == "__main__":
    _preprocess_data(sys.argv[1])
    print("Completed")
