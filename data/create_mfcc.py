#!/usr/bin/env python

import sys
import os
from os import path
import fnmatch
sys.path.insert(0,'..')
from utils import featurize_mfcc, featurize_spectogram
import numpy as np

def process(args):
    datapath = args
    target = path.join(datapath, "TIMIT")
    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*_rif.wav"):
            full_wav = os.path.join(root, filename)
            mfcc_file = full_wav[:-8] + ".mfcc.npy"
            print(mfcc_file)
            # MFCC
            mfcc_features = featurize_mfcc(full_wav)
            np.save(mfcc_file, mfcc_features)

if __name__ == "__main__":
    process(sys.argv[1])
    print("Completed")