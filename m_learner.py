from numpy.lib.function_base import average
import data_manip
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


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
    X_test, Y_test= data_manip.read_indexed("./data/test_with_label_"+str(ver)+".csv")
    dtc = tree.DecisionTreeClassifier()
    dtc.max_depth=85
    dtc.criterion="entropy"
    dtc.min_samples_split=5
    dtc.min_impurity_decrease=0.00025
    dtc.class_weight=None
    dtc.fit(X,Y)
    output_arr = []
    for i in range(0,len(X_test)) :
        output_arr.append(dtc.predict([X_test[i]])[0])
    calculate_metrics(Y_test,output_arr)
    data_manip.write_indexed("./output/Best-DT-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))

def calculate_metrics(actual,expected):
    print(accuracy_score(actual,expected))
    print(recall_score(actual,expected,average='weighted'))
    print(precision_score(actual,expected,average='weighted'))
    print(f1_score(actual,expected,average='weighted'))
    print(confusion_matrix(actual,expected))


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

def best_multi_layered_perceptron(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test = data_manip.read_unindexed("./data/test_no_label_"+str(ver)+".csv")
    mlp = MLPClassifier()
    parameters = {
        'hidden_layer_sizes': [(30, 50), (10, 10 ,10)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam']
    }   
    gridMLP = GridSearchCV(mlp, parameters)
    gridMLP.fit(X, Y)
    output_arr = [gridMLP.predict([X])[0] for X in X_test]
    data_manip.write_indexed("./output/Best-MLP-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))
