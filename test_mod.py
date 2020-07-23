import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch.nn
import torchvision
from imutils import face_utils
from sklearn import svm
from tqdm import tqdm
from PIL import Image
import glob
import os
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import save, load
import os.path
from tabulate import tabulate

namedict = {'andrew yang':0, 'barack obama':1, 'bernie sanders':2, 'joe biden':3, 'lilly singh':4, 'malala yousafzai': 5, 'michelle obama':6, 'ramy youssef':7, 'trevor noah':8, 'unknown_class':9}

def accuracy_metrics(truth_labels, tested_labels):
    p_same = len(truth_labels)
    TP = truth_labels == tested_labels
    FP = len(tested_labels) - len(TP)
    TN = [tested_labels not in TP]
    FN =  [tested_labels not in truth_labels] #we got it wrong

    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP)
    negative_pred_val = TN/(TN+FN)
    metrics = [sensitivity, specificity, accuracy, precision, negative_pred_val]
    return metrics


def plot_acc(testname, data, mode = "test", note = ""):
    file1 = open("data/test/test_results/" + testname + " test results.txt", "a+")  # append mode
    file1.write(mode + 'results for ' + testname + datetime.datetime.now().strftime(" on %Y-%m-%d %H:%M:%S") +"\n")
    file1.write("Accuracy: \n")
    onehot = np.zeros((10,1))
    for name in set(data):
        acc = data.count(name)/len(data)*100
        file1.write("   "+name + " {}%\n".format(acc))
        onehot[namedict[name]] = acc

    file1.write("Number of frames: {}\n".format(len(data))) 
    file1.write("Note: "+ note+"\n\n") 
    test_res_l = max(len(data), 1)
    print("Accuracy is {}%".format(data.count(testname)/test_res_l*100))
    file1.close()
    # Save result
    potfile = "data/test/test_results/accs/" + testname + ".npy"
    files = glob.glob('data/embeddings/live/*.npy')
    data = np.asarray([])
    if os.path.isfile(potfile):
        data = load("data/test/test_results/accs/" + testname + ".npy")
    onehot = np.vstack([data, onehot])
    print(onehot)
    save("data/test/test_results/accs/" + testname + ".npy", data) 

def plot():

    dat_dict = {}
    for file in glob.glob("data/test/test_results/accs/*.npy"):
        name = (file.split("/")[-1].replace("_", " ")).split(".")[0]
        dat_dict[name] = load(file)
    # dat_dict['x'] = np.asarray(range(len(data)))
    # print("!! dat_dict: ", dat_dict)

    # df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21) })
    # df=pd.DataFrame(dat_dict)
    # print(list(dat_dict.values()))
    # print(list(dat_dict.keys()))
    # plt.plot(list(dat_dict.values())[0], label=list(dat_dict.keys()))
    keys = list(dat_dict.keys())
    vals = list(dat_dict.values())

    df = pd.DataFrame(dat_dict) 
    file1 = open("data/test/test_results/table_results.txt", "a+")  # append mode
    file1.write(tabulate(df, headers='keys', tablefmt='psql'))
    file1.close()

    for i in range(len(keys)):
        # print(vals[i])
        # print(keys[i])
        plt.plot(vals[i], label = keys[i], marker='o', markerfacecolor='blue', markersize=5)
        val = vals[i]
        for i in range(len(val)):
            plt.annotate("{:.2f}".format(val[i]),(i,val[i]), textcoords="offset points", xytext=(0,10), ha='center')
    # for col in dat_dict.keys():
    #     print(col)
    #     plt.plot(  col, data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

    # # multiple line plot
    # plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
    # plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
    # plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")

    plt.legend()
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracies throughout training and testing')
    plt.xticks(np.arange(0,3,1),labels = ["train", "dev", "test"])
    plt.grid()
    plt.show()
    plt.savefig('test.png')
    # p_same = len(truth_labels)
    # true_accepts = truth_labels == tested_labels
    # p_diff = len(tested_labels) - len(true_accepts)
    # false_accepts= [tested_labels not in true_accepts]
    # validation_rate = true_accepts/p_same
    # false_accept_rate =  false_accepts/p_diff



    # return validation_rate, false_accept_rate

# plot()