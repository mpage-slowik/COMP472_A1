import data_manip
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


def gnb_predictor(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    gnb = GaussianNB()
    X_test = data_manip.read_unindexed("./data/test_no_label_"+str(ver)+".csv")
    gnb.fit(X,Y)
    output_arr = []
    for i in range(0,len(X_test)) :
        output_arr.append(gnb.predict([X_test[i]])[0])
    data_manip.write_indexed("./output/GNB-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))
