import fnmatch
import os
import pandas as pd
import char_map
from utils import text_to_int_sequence
from char_map import get_number_of_char_classes

#######################################################

def combine_all_wavs_and_trans_from_csvs(csvslist):
    ''' Load data CSV 
    '''
    df_all = pd.DataFrame()
    for csv in csvslist.split(','):
        print("Reading csv:",csv)
        if os.path.isfile(csv):
            try:
                df_new = pd.read_csv(csv, sep=',', encoding='ascii')
            except:
                print("NOT - ASCII, use UTF-8")
                df_new = pd.read_csv(csv, sep=',', encoding='utf-8')
                # df_new.transcript.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
            df_all = df_all.append(df_new)
    print("Finished reading in data")
    df_final = df_all
    listcomb = df_all['transcript'].tolist()
    comb = []
    for t in listcomb:
        #s = t.decode('utf-8').encode('ascii', errors='ignore')
        comb.append(' '.join(t.split()))
    ## SIZE CHECKS
    num_classes = get_number_of_char_classes()
    print("Number of classes:", num_classes)
    dataproperties = {
        'target': "timit",
        'num_classes': num_classes,
    }
    #remove mem
    del df_all
    del listcomb
    return dataproperties, df_final
