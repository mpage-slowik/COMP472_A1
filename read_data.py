import pandas as pd

def read_indexed(filename):
    df_data = pd.read_csv(filename,delimiter=',',header=None,usecols=range(0,1024))
    df_index = pd.read_csv(filename,delimiter=',',header=None,usecols=[1024])
    X = df_data.to_numpy()
    Y = df_index.to_numpy()
    return X,Y

def read_unindexed(filename):
    df_data = pd.read_csv(filename,delimiter=',',header=None,usecols=range(0,1024))
    X = df_data.to_numpy()
    return X
