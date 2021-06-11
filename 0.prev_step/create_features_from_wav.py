#!/usr/bin/env python

import sys
sys.path.insert(0,'..')
from utils import featurize_mfcc, featurize_spectogram
import numpy as np

def process(path):
    # MFCC
    mfcc_features = featurize_mfcc(path)
    np.save(path + ".mfcc.npy", mfcc_features)
    np.savetxt(path + ".mfcc.csv", mfcc_features)
    print(type(mfcc_features))
    print(mfcc_features.shape)
    
    # Spectogram
    spectogram_features = featurize_spectogram(path)
    np.save(path + ".spectogram.npy", spectogram_features)
    np.savetxt(path + ".spectogram.csv", spectogram_features)
    print(type(spectogram_features))
    print(spectogram_features.shape)

if __name__ == "__main__":
    process(sys.argv[1])