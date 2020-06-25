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

from align_faces import extract_faces, align_faces
from dataset import FaceDataset
from openface import load_openface, preprocess_batch
from classifiers.binary_face_classifier import BinaryFaceClassifier, BinaryFaceNetwork
from repl import recognize

def test():
    files = glob.glob("data/test/test_videos/*/*")
    # recognize()
    # return testfile, testname

def accuracy(truth_labels, tested_labels):
    p_same = len(truth_labels)
    true_accepts = truth_labels == tested_labels
    p_diff = len(tested_labels) - len(true_accepts)
    false_accepts= [tested_labels not in true_accepts]
    validation_rate = true_accepts/p_same
    false_accept_rate =  false_accepts/p_diff
    return validation_rate, false_accept_rate
