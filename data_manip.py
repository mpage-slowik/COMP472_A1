import numpy as np
import pandas as pd

def read_indexed(filename):
    df_data = pd.read_csv(filename,header=None,usecols=range(0,1024))
    df_index = pd.read_csv(filename,header=None,usecols=[1024])
    X = df_data.to_numpy()
    Y = df_index.to_numpy().flatten()
    return X,Y

def read_unindexed(filename):
    df_data = pd.read_csv(filename,delimiter=',',header=None,usecols=range(0,1024))
    
    X = df_data.to_numpy()
    return X

def write_indexed(filename, df):
    df.to_csv(filename, header=None)

def add_performance(filename, data, perf_type=""):
    with open(filename,mode='a') as f:
        if isinstance(data,float):
            f.write(perf_type + ',' + str(data) + '\n')
        else:
            for index in range(0,len(data)):
                f.write(str(index) + ',' + str(data[index]) + '\n')

def add_class_measures(filename,data):
    with open(filename,mode='a') as f:
        f.write('class,precision,recall,fscore,total\n')
        for index in range(0,len(data[0])):
            f.write(str(index) + ',' + str(data[0][index])+ ',' + str(data[1][index])+ ',' + str(data[2][index])+ ',' + str(data[3][index])+'\n')
   
