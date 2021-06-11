# Speech To Phoneme Project

This project is developed to create a Deep Learning algorithm able to distinguish English mispronunciations. The idea behind this is to create a mobile app to help people to pronounce English correctly.

## Installation

### Requirements

    $ pip3 install -r requirements.txt 

Installed components:

- pandas==0.20.3
- SoundFile==0.10.3.post1
- matplotlib==2.1.1
- tensorflow==1.6.0
- Keras==2.2.5
- h5py==2.7.1
- numpy==1.13.3
- python_speech_features

### Import data

1. Uncompress TIMIT.zip onto `./data` directory. The TIMIT/TRAIN directory must be on this subrirectory.
2. Change directori to data: `cd data`
3. Run `python3 import_timit_phoneme.py .`
4. Convert wav files into the correct format: `python3 prepare_wav.py .`
5. Create mfcc files with `python3 create_mfcc.py .`

### Run with

Sample run:

    $ python3 run-train.py --train_files=./data/TIMIT/sample.csv --valid_files=./data/TIMIT/sample.csv --batchsize=2

Bolt run:

    $ python3 run-train.py --train_files=./data/TIMIT/timit_phoneme_train.csv --valid_files=./data/TIMIT/timit_phoneme_test.csv --epochs=200 --fc_size=512

## Watch results

Run Tensorboard:

    $ tensorboard --logdir=/full_path_to_your_logs