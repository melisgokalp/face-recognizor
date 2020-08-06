import argparse
# import time
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
import pandas as pd
from numpy import save, load
from tabulate import tabulate
from dataset import FaceDataset

import pandas as pd
import matplotlib.pyplot as plt

# roc curve and auc score
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc


def plot_roc_curve(fpr, tpr, fpr1=None,tpr1= None):
    dauc = auc(fpr, tpr)
    print("auc is " + str(dauc))
    plt.plot(fpr, tpr, color='orange', label='Dlib AUC=' + str(dauc))

    dauc = auc(fpr1, tpr1)
    print("auc is " + str(dauc))
    plt.plot(fpr1, tpr1, color='blue', label='OpenFace AUC=' + str(dauc))

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()



def svm_unknown_classes():
    N_ITERS = 10
    t = []
    f = []
    mode = "comp"
    mode1 = "dlibcomp"
    for _ in tqdm(range(N_ITERS), total=N_ITERS):
        known = FaceDataset("data/embeddings/"+mode+"/test", "data/embeddings/"+mode+"/test", n=100)
        known_train, known_labels = known.train()
        known_test, _ = known.test()
        unknown = FaceDataset("data/embeddings/"+mode+"/dev",  "data/embeddings/"+mode+"/dev", n=100)
        unknown_data, _, _ = unknown.all()
        seed = FaceDataset("data/embeddings/"+mode+"/train",  "data/embeddings/"+mode+"/train", n=100)
        seed_train, seed_labels, _ = seed.all()
        # assign our unknown class to be 0 and increment all the labels in known by 1
        known_labels = known_labels + 1
        seed_labels = np.zeros(len(seed_labels))

        # train the SVM on the classes with the random seed
        full_training = np.concatenate([known_train, seed_train])
        full_labels = np.concatenate([known_labels, seed_labels])
        clf = svm.SVC(kernel="rbf", gamma="scale", C=1.0, probability=True)
        clf.fit(full_training, full_labels)

        # run SVM on the unknown set
        unknown_probs = clf.predict_proba(unknown_data)
        pred = np.argmax(unknown_probs, axis=1)
        unknown_confs = np.max(unknown_probs, axis=1)
        unknown_confs[pred == 0] = 0

        # run SVM on known set
        known_probs = clf.predict_proba(known_test)
        pred = np.argmax(known_probs, axis=1)
        known_confs = np.max(known_probs, axis=1)
        known_confs[pred == 0] = 0

        # TPR = rate of unknown faces correctly qualified as so
        # FPR = rate of known faces being qualified as unknown faces
        TPRs = []
        FPRs = []
        thresholds = np.linspace(0, 1, 1000)
        for x in thresholds:
            TPR = np.mean(unknown_confs < x)
            FPR = np.mean(known_confs < x)
            TPRs.append(TPR)
            FPRs.append(FPR)

        t.append(TPRs)
        f.append(FPRs)

    t = np.mean(t, axis=0)
    f = np.mean(f, axis=0)
    np.save(mode + "_10iter" + "_tpr.npy", t)
    np.save(mode + "_10iter" + "_fpr.npy", f)
    # roc_auc = auc(FPRs, TPRs)
    # plot_roc_curve(f, t)
    print(t.shape)
    print(f.shape)
    
# mode = "comp"
# mode1 = "dlibcomp"
# t = np.load(mode1 + "_10iter" + "_tpr.npy")
# f= np.load(mode1 + "_10iter" + "_fpr.npy")
# t1 = np.load(mode + "_10iter" + "_tpr.npy")
# f1= np.load(mode + "_10iter" + "_fpr.npy")
# plot_roc_curve(f, t,f1,t1)

# clf = svm.SVC(kernel="rbf", C=1.0, probability=True)
# ds = FaceDataset("data/embeddings/live", "data/embeddings/train")
# data, labels, idx_to_name = ds.all()
# # Make test embeddings
# print(data.shape)
# print(labels.shape)
# probas_ = clf.fit(data[:1000], labels[:1000])
# # predict classes for all faces and label them if greater than threshold
# probs = clf.predict_proba(data)
# unknown_class_prob = probs[0][-1]
# print(probs.shape)
# predictions = np.argmax(probs, axis=1)
# probs = np.max(probs, axis=1)
# print(predictions)


def accuracy_metrics(truth_labels, tested_labels):
    # np.save("data/test/test_results/accs/" + "tested_labels" + ".npy", np.asarray(tested_labels)) 
    # np.save("data/test/test_results/accs/" + "truth_labels" + ".npy", np.asarray(truth_labels)) 
    # print(len(truth_labels))
    # print(len(tested_labels[1]))
    truth_labels= np.load("data/test/test_results/accs/" + "truth_labels" + ".npy") 
    tested_labels= np.load("data/test/test_results/accs/" + "tested_labels" + ".npy")[1]

    for i in range(len(truth_labels)):
        name = truth_labels[i].split("/")[-2]
        truth_labels[i] = name

    res =  [x==y for x, y in zip(tested_labels, truth_labels)]

    test_elements = list(set(tested_labels))
    for e in test_elements:
        if e[:2] == "n0":
            test_subject = e
    # print(sum(match))
    # p_same = len(truth_labels)
    TP = res.count(True)
    FP = max(0,tested_labels.count(test_subject) - list(truth_labels).count(test_subject))
    test_not_test_subject = len(tested_labels) - tested_labels.count(test_subject)
    truth_not_test_subject = len(truth_labels) - list(truth_labels).count(test_subject)
    FN = test_not_test_subject - truth_not_test_subject #we got it wrong
    TN = res.count(False) - FN

    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    negative_pred_val = TN/(TN+FN)

    f_score = (2 * recall * precision) / (recall + precision)
    metrics = [sensitivity, specificity, accuracy, precision, negative_pred_val, f_score]
    print(metrics)
    # plot_roc_curve(FP/, precisio)
    return metrics


def plot_acc(data, mode = "test", note = ""):
    # We save names of all the embeddings
    # live_embeddings = glob.glob('data/embeddings/live/*.npy')
    # all_names = []
    # for embedding in live_embeddings:
    #     name = embedding.split("/")[-1]
    #     name = name.split(".")[0]
    #     all_names.append(name)
    # all_names.append("unknown")
    # all_names.append("no detection")
    # print(all_names)
    # onehot = np.zeros((len(all_names),1))
    # for name in data:
    #     onehot[all_names.index(name)] = 1

    onehot = np.asarray(data)
    print(data)
    for name in data:
        onehot[all_names.index(name)] = 1

    file_name = "result_data"
    result_data = "data/test/test_results/accs/" + file_name + ".npy"
    data = np.asarray([])
    if os.path.isfile(result_data):
        data = load(result_data)
    onehot = np.vstack([data, onehot])
    print(onehot)
    save(result_data, data)

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
accuracy_metrics("","")

# svm_unknown_classes()