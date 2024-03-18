
import os
import pandas

from utils import read_file

def ingest(dataset):

    path = 'data/' + dataset + '/'
    explicit_split = False

    flist = os.listdir(path)
    train_flist = [f for f in os.listdir(path) if 'train' in f.lower()]
    test_flist = [f for f in os.listdir(path) if 'test' in f.lower()]

    #if both train file list and test file list empty or one and only one between train_flist and test_flist has at least one element
    #we consider explicit split
    explicit_split = (len(train_flist) == 0 and len(test_flist) == 0) or (len(train_flist) > 0 ^ len(test_flist) > 0)
    if(explicit_split): print("[DEBUG] Explicit split will be performed...")
    else: print("[DEBUG] Split has been already determined by files partition...")

    #Establishing priority of CSV over PARQUET format
    if(explicit_split):
        for f in flist:
            if('.csv' in f):
                file_format = "csv"
                break
            else: file_format = "parquet"
        
        #file_format = flist[0][flist[0].find('.') + 1:]

    else:
        file_format = train_flist[0][train_flist[0].find('.') + 1:]
    
    if(dataset == "NSL-KDD"): file_format = 'csv'           #NSL-KDD has .csv files masked behind .txt
            
    print("[DEBUG] File format detected: " + file_format)

    #Determining which files compose the data set (for explicit split)
    if(explicit_split):
        df_flist = []           #list of files involved in Pandas DataFrame creation
        for f in flist:
            if("." + file_format in f): df_flist.append(f)

    #Building of Pandas DataFrames which will contain all the info
    df_list = []            #list of Pandas DataFrame to pass to the next step
    if(explicit_split):
        print("[DEBUG] Reading file '" + df_flist[0] + "'...")
        df0, features_list = read_file(path + df_flist[0], file_format)
        for file in df_flist[1:]:
            print("[DEBUG] Reading file '" + file + "'...")
            df, features_list = read_file(path + file, file_format)
            df0 = pandas.concat([df0, df])

        df_list.append(df0)

    else:
        df_tr0, features_list = read_file(path + train_flist[0], file_format)
        df_te0, features_list = read_file(path + test_flist[0], file_format)
        for file in train_flist[1:]:
            df, features_list = read_file(path + file, file_format)
            df_tr0 = pandas.concat([df_tr0, df])

        df_list.append(df_tr0)

        for file in test_flist[1:]:
            df, features_list = read_file(path + file, file_format)
            df_te0 = pandas.concat([df_te0, df])

        df_list.append(df_te0)

    return df_list, features_list