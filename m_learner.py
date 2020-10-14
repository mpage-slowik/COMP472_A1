from matplotlib.pyplot import yticks
from numpy.core.numeric import zeros_like
from numpy.lib.function_base import average
import data_manip
import os
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_fscore_support 


def gnb_predictor(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test, Y_test = data_manip.read_indexed("./data/test_with_label_"+str(ver)+".csv")
    gnb = GaussianNB()
    gnb.fit(X,Y)
    output_arr = [gnb.predict([X])[0] for X in X_test]
    data_manip.write_indexed("./output/GNB-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))
    calculate_metrics(Y_test,output_arr,ver,"./output/GNB-DS"+str(ver)+".csv")


def base_dt(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test, Y_test = data_manip.read_indexed("./data/test_with_label_"+str(ver)+".csv")
    dtc = tree.DecisionTreeClassifier()
    dtc.fit(X,Y)
    output_arr = [dtc.predict([X])[0] for X in X_test]
    data_manip.write_indexed("./output/Base-DT-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))
    calculate_metrics(Y_test,output_arr,ver,"./output/Base-DT-DS"+str(ver)+".csv")


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
    output_arr = [dtc.predict([X])[0] for X in X_test]
    data_manip.write_indexed("./output/Best-DT-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))
    calculate_metrics(Y_test,output_arr,ver,"./output/Best-DT-DS"+str(ver)+".csv")


def calculate_metrics(expected,actual,ver,fname):
    if ver == 1:
        data_manip.add_performance(fname,confusion_matrix(expected,actual,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]))
    else:
        data_manip.add_performance(fname,confusion_matrix(expected,actual,labels=[0,1,2,3,4,5,6,7,8,9]))


    # print(recall_score(actual,expected,average='weighted'))
    # print(precision_score(actual,expected,average='weighted'))
    if ver == 1:
        data_manip.add_class_measures(fname,precision_recall_fscore_support(expected,actual,average=None,zero_division=0,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]))
    else:
        data_manip.add_class_measures(fname,precision_recall_fscore_support(expected,actual,average=None,zero_division=0,labels=[0,1,2,3,4,5,6,7,8,9]))


    data_manip.add_performance(fname,accuracy_score(expected,actual),'accuracy')
    data_manip.add_performance(fname,f1_score(expected,actual,average='macro'),'f1_macro')
    data_manip.add_performance(fname,f1_score(expected,actual,average='weighted'),'f1_weighted')


def calculate_distribution(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    df = pd.DataFrame({'index':Y})
    return df['index'].value_counts()

def default_perceptron(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test, Y_test = data_manip.read_indexed("./data/test_with_label_"+str(ver)+".csv")
    per = Perceptron() # Default params
    per.fit(X, Y)
    output_arr = [per.predict([X])[0] for X in X_test]
    data_manip.write_indexed("./output/PER-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))
    calculate_metrics(Y_test,output_arr,ver,"./output/PER-DS"+str(ver)+".csv")

def base_multi_layered_perceptron(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test, Y_test = data_manip.read_indexed("./data/test_with_label_"+str(ver)+".csv")
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation="logistic", solver='sgd',max_iter=400)
    mlp.fit(X, Y)
    output_arr = [mlp.predict([X])[0] for X in X_test]
    data_manip.write_indexed("./output/Base-MLP-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))
    calculate_metrics(Y_test,output_arr,ver,"./output/Base-MLP-DS"+str(ver)+".csv")

def best_multi_layered_perceptron(ver):
    X,Y = data_manip.read_indexed("./data/train_"+str(ver)+".csv")
    X_test, Y_test = data_manip.read_indexed("./data/test_with_label_"+str(ver)+".csv")
    mlp = MLPClassifier(max_iter=400)
    parameters = {
        'hidden_layer_sizes': [(30,30,30) ,(100,)],
        'activation': ['relu'],
        'solver': ['sgd']
    }   
    gridMLP = GridSearchCV(mlp, parameters)
    gridMLP.fit(X, Y)
    output_arr = [gridMLP.predict([X])[0] for X in X_test]
    data_manip.write_indexed("./output/Best-MLP-DS"+str(ver)+".csv",pd.DataFrame({'index':output_arr}))
    calculate_metrics(Y_test,output_arr,ver,"./output/Best-MLP-DS"+str(ver)+".csv")

def plot_lib():
    folder = "./plot_data"
    # ax = fig.add_subplot(1,1,1)
    bar_width=0.15
    pos2 = 0.0
    pos = 0.0


    for file in os.listdir(folder):
        df = pd.read_csv(os.path.join(folder,file))

        if '1' in file:
            plot_bars(1,df['fscore'],pos,bar_width,df['class'],file.split('.')[0])
            pos+=(bar_width)

            # plt.figure(1, (10,100))
            # df['precision'] = pd.Series([(val*100) for val in df['precision']])
            # plt.xticks(np.arange(0,26,1))
            # plt.yticks(np.arange(0,110,10))
            # plt.plot(df['precision'])
        else:
            plot_bars(2,df['fscore'],pos2,bar_width,df['class'],file.split('.')[0])
            pos2+=(bar_width)

            # plt.figure(2, (10,100))
            # plt.xticks(np.arange(0,10,1))
            # plt.yticks(np.arange(0,110,10))
            # plt.bar(df['class'],df['precision'],label=file)#.plot(df['precision'])

    plt.show()

def plot_bars(ver,data,pos,bar_width,labels,algo_name):
        initial = np.arange(len(labels))
        fig = plt.figure(ver,(20,100))
        ax = plt.subplot(111)
        title = data.name
        data= pd.Series([(val*100) for val in data])
        ax.bar(initial+pos,data,width=bar_width,label=algo_name)
        ax.set_xticks(initial+(3*bar_width))
        ax.set_xticklabels(labels)
        ax.set_title(title)
        plt.yticks(np.arange(0,105,5))
        ml = MultipleLocator(5)
        ax.yaxis.set_minor_locator(ml)
        # plt.minorticks_on()
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.legend()


