#!/usr/bin/env python

import os
from os import path
import sys
import tarfile
import fnmatch
import subprocess

def _preprocess_data(args):
    datapath = args
    target = path.join(datapath, "TIMIT")
    print("Preprocessing data")
    # We convert the .WAV (NIST sphere format) into MSOFT .wav
    # creates _rif.wav as the new .wav file
    for root, dirnames, filenames in os.walk(target):
        for filename in fnmatch.filter(filenames, "*.WAV"):
            sph_file = os.path.join(root, filename)
            wav_file = os.path.join(root, filename)[:-4] + "_rif.wav"
            print("converting {} to {}".format(sph_file, wav_file))
            subprocess.check_call(["sox", sph_file, wav_file])

if __name__ == "__main__":
    _preprocess_data(sys.argv[1])
    print("Completed")