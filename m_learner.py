import data_manip
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


def gnb_predictor(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test = data_manip.read_unindexed("./data/test_no_label_"+str(ver)+".csv")
    gnb = GaussianNB()
    gnb.fit(X,Y)
    output_arr = []
    for i in range(0,len(X_test)) :
        output_arr.append(gnb.predict([X_test[i]])[0])
    data_manip.write_indexed("./output/GNB-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))

def base_dt(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test = data_manip.read_unindexed("./data/test_no_label_"+str(ver)+".csv")
    dtc = tree.DecisionTreeClassifier()
    dtc.fit(X,Y)
    output_arr = []
    for i in range(0,len(X_test)) :
        output_arr.append(dtc.predict([X_test[i]])[0])
    data_manip.write_indexed("./output/Base-DT-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))

def best_dt(ver):
    """
    splitting criterion: gini and entropy
    maximum depth of the tree: 10 and no maximum
    minimum number of samples to split an internal node: experiment with values of your choice
    minimum impurity decrease: experiment with values of your choice
    class weight: None and balanced
    """
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test = data_manip.read_unindexed("./data/test_no_label_"+str(ver)+".csv")
    dtc = tree.DecisionTreeClassifier()
    dtc.max_depth=10
    dtc.criterion="entropy"
    # dtc.min_samples_split=4
    # dtc.min_impurity_decrease=2.0
    # dtc.class_weight="balanced"
    dtc.fit(X,Y)
    output_arr = []
    for i in range(0,len(X_test)) :
        output_arr.append(dtc.predict([X_test[i]])[0])
    data_manip.write_indexed("./output/Best-DT-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))

def calculate_distribution(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    df = pd.DataFrame({'index':Y})
    return df['index'].value_counts()

def default_perceptron(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test = data_manip.read_unindexed("./data/test_no_label_"+str(ver)+".csv")
    per = Perceptron() # Default params
    per.fit(X, Y)
    output_arr = [per.predict([X])[0] for X in X_test]
    data_manip.write_indexed("./output/PER-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))

def base_multi_layered_perceptron(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test = data_manip.read_unindexed("./data/test_no_label_"+str(ver)+".csv")
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver='sgd')
    mlp.fit(X, Y)
    output_arr = [mlp.predict([X])[0] for X in X_test]
    data_manip.write_indexed("./output/Base-MLP-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))