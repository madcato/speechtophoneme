#!/usr/bin/env python

import argparse
from data import combine_all_wavs_and_trans_from_csvs
import timeit
import datetime
import numpy as np

class Timer:
    """Measure time used."""
    # Ref: https://stackoverflow.com/a/57931660/

    def __init__(self, round_ndigits: int = 0):
        self._round_ndigits = round_ndigits
        self._start_time = timeit.default_timer()

    def __call__(self) -> float:
        return timeit.default_timer() - self._start_time

    def __str__(self) -> str:
        return str(datetime.timedelta(seconds=round(self(), self._round_ndigits)))

def load(filename):
    return np.load(filename)

def main(args):
    train_dataprops, df_train = combine_all_wavs_and_trans_from_csvs(args.train_files)
    labels = df_train['transcript']
    ############ 
    # print("Benchmark1 start")
    # timer = Timer()
    # for i, ele in enumerate(labels):
    #     vec = eval(ele)
    #     # print(vec)
    # print(f'Time elapsed is {timer}.')
    # print("Benchmark1 stop")
    ##############
    ############ 
    print("Benchmark2 start")
    timer2 = Timer()
    for i, filename in enumerate(labels):
        vec = load(filename)
    print(f'Time elapsed is {timer2}.')
    print("Benchmark2 stop")
    ##############

if __name__ == '__main__':
    print("Getting args")
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_files', type=str, default='',
                       help='list of all train files, seperated by a comma if multiple')
    parser.add_argument('--sortagrad', type=bool, default=True,
                       help='If true, we sort utterances by their length in the first epoch')
    args = parser.parse_args()
    print(args)
    main(args)
    