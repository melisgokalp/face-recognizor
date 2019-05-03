import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch.nn
from imutils import face_utils
from sklearn import svm
from tqdm import tqdm

from align_faces import extract_faces, align_faces
from dataset import FaceDataset
from openface import load_openface, preprocess_batch

CONF_THRESHOLD = 0.8
CONF_TO_STORE = 30


def capture_faces(seconds=10, sampling_duration=0.1, debug=False):
    print("Capturing! about to capture {} seconds of video".format(seconds))
    start_time = time.time()

    # face_locs stores the bounding box coordinates
    face_locs = []
    # frames stores the actual images
    frames = []

    while time.time() - start_time < seconds:
        ret, frame = video_capture.read()
        if ret:
            faces = extract_faces(frame)
            if len(faces) == 1:
                frames.append(frame)
                face_locs.append(faces[0])
                print("Taken sample.")

            if len(faces) == 0:
                print("No faces found.")

            if len(faces) > 1:
                print("We have found {} faces, and there should only be one".format(len(faces)))
        else:
            print("ERROR: No sample taken")
        # lock the loop to system time
        time.sleep(sampling_duration - ((time.time() - start_time) % sampling_duration))

    # extract the faces afterwards
    print("Extracting faces from samples")
    samples = []
    for i in tqdm(range(len(face_locs)), total=len(face_locs)):
        rect = face_locs[i]
        frame = frames[i]
        sample = align_faces(frame, [rect])[0]
        samples.append(sample)
        if debug:
            cv2.imshow("samples", sample)
            cv2.waitKey(0)

    return samples


def add_name_to_dictionary(name, num_classes):
    idx_to_name[num_classes] = name
    # TODO: with later models, implement persistence as below
    # f = open("data/idx_to_name.pkl", 'w')
    # pickle.dump(idx_to_name, f)


def retrain_classifier(clf):
    ds = FaceDataset("data/embeddings")
    data, labels = ds.all()
    clf = clf.fit(data, labels)
    return clf


def add_face(clf, num_classes):
    name = input("We don't recognize you! Please enter your name:\n").strip().lower()
    while name in name_to_idx:
        name = input("We don't recognize you! Please enter your name:\n").strip().lower()
    samples = capture_faces()
    while len(samples) < 75:
        print("We could not capture sufficient samples. Please try again.\n")
        samples = capture_faces()
    embeddings = preprocess_batch(samples)
    embeddings = openFace(embeddings)
    embeddings = embeddings.detach().numpy()

    # save name and embeddings
    np.save("data/embeddings/{}.npy".format(name), embeddings)
    add_name_to_dictionary(name, num_classes)
    clf = retrain_classifier(clf)
    return clf


def load_model():
    # TODO: in the future we should look at model persistence to disk
    clf = svm.SVC(kernel="linear", C=1.6, probability=True)
    ds = FaceDataset("data/embeddings")
    data, labels = ds.all()
    num_classes = len(np.unique(labels))
    clf = clf.fit(data, labels)
    return clf, num_classes, ds.ix_to_name

def main(clf, num_classes):
    # to store previous confidences to determine whether a face exists
    prev_conf = deque(maxlen=CONF_TO_STORE)
    print("Starting...")
    while True:
        # ret is error code but we don't care about it
        ret, frame = video_capture.read()
        if ret:
            # extract and align faces
            rects = extract_faces(frame)
            if len(rects) > 0:

                # draw all bounding boxes for faces
                for i in range(len(rects)):
                    x, y, w, h = face_utils.rect_to_bb(rects[i])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # generate embeddings
                faces = align_faces(frame, rects)
                tensor = preprocess_batch(faces)
                embeddings = openFace(tensor)
                embeddings = embeddings.detach().numpy()

                # predict classes for all faces and label them if greater than threshold
                probs = clf.predict_proba(embeddings)
                predictions = np.argmax(probs, axis=1)
                probs = np.max(probs, axis=1)
                print(probs)
                names = [idx_to_name[idx] for idx in predictions]
                # replace all faces below confidence w unknown
                names = [names[i] if probs[i] > CONF_THRESHOLD else "UNKNOWN" for i in range(len(probs))]
                print("Hi {}!".format(names))
                for i in range(len(names)):
                    x, y, w, h = face_utils.rect_to_bb(rects[i])
                    cv2.putText(frame, names[i], (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

                # determine if we need to trigger retraining
                # we only retrain if there is one person in the frame and they are unrecognized or there are 0 classes
                if len(faces) == 1:
                    prev_conf.append(probs[0])
                    if np.mean(prev_conf) < CONF_THRESHOLD and len(prev_conf) == CONF_TO_STORE:
                        clf = add_face(clf, num_classes)
                        num_classes += 1
                        prev_conf.clear()
            else:
                print("No faces detected.")
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return
        else:
            print("ERROR: no frame captured")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="run with this flag to run on a GPU")
    args = vars(parser.parse_args())

    device = torch.device("cuda") if args["gpu"] and torch.cuda.is_available() else torch.device("cpu")
    print("Using device {}".format(device))

    video_capture = cv2.VideoCapture(0)
    openFace = load_openface(device)

    # samples = capture_faces()
    # embeddings = preprocess_batch(samples)
    # embeddings = openFace(embeddings)
    # embeddings = embeddings.detach().numpy()
    # 
    # # save name and embeddings
    # np.save("data/embeddings/varun.npy", embeddings)

    clf, num_classes, idx_to_name = load_model()
    # cannot function as a classifier if less than 2 classes
    assert num_classes >= 2
    name_to_idx = {idx_to_name[idx]: idx for idx in idx_to_name}
    main(clf, num_classes)

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()