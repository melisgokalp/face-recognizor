import argparse
import os, time, datetime
from collections import deque
import cv2
import numpy as np
import torch.nn
import torchvision
from imutils import face_utils
from sklearn import svm
from tqdm import tqdm
from PIL import Image
import test_mod as tmod
import glob
import random
from align_faces import extract_faces, align_faces
from dataset import FaceDataset
from openface import load_openface, preprocess_batch
from classifiers.binary_face_classifier import BinaryFaceClassifier, BinaryFaceNetwork

CONF_THRESHOLD = 0.6
CONF_TO_STORE = 30
LIVE_EMBEDDINGS = "data/embeddings/live"
TRAIN_EMBEDDINGS = "data/embeddings/train"

def capture_faces(seconds=10, sampling_duration=0.1, debug=False):
    print("Capturing! about to capture {} seconds of video".format(seconds))
    start_time = time.time()

    # face_locs stores the bounding box coordinates
    face_locs = []
    # frames stores the actual images
    frames = []
    ctr = 1
    while time.time() - start_time < seconds:
        ret, frame = video_capture.read()
        if ret:
            faces = extract_faces(frame)
            if len(faces) == 1:
                frames.append(frame)
                face_locs.append(faces[0])
                print("Took sample: " + str(ctr))
                ctr+=1

            if len(faces) == 0:
                print("No faces found.")

            if len(faces) > 1:
                print("We have found {} faces, and there should only be one".format(len(faces)))
        else:
            print("ERROR: No sample taken")
        # lock the loop to system time
        time.sleep(sampling_duration - ((time.time() - start_time) % sampling_duration))

    # extract the faces afterwards
    print("Extracting faces from samples\n")
    samples = []
    for i in tqdm(range(len(face_locs)), total=len(face_locs)):
        rect = face_locs[i]
        frame = frames[i]
        sample = align_faces(frame, [rect])[0]
        samples.append(sample)
        data_aug = augment_data(sample)
        samples.extend(data_aug)
        if debug:
            cv2.imshow("samples", sample)
            cv2.waitKey(0)

    return samples

# for data augmentation — adjusts hue and saturation
def augment_data(image):
    img = Image.fromarray(image)
    hue1 = torchvision.transforms.functional.adjust_hue(img, .05)
    hue2 = torchvision.transforms.functional.adjust_hue(img, -.05)
    sat1 = torchvision.transforms.functional.adjust_saturation(img, 1.35)
    sat2 = torchvision.transforms.functional.adjust_saturation(img, .65)
    return [np.array(hue1), np.array(hue2), np.array(sat1), np.array(sat2)]

def retrain_classifier(clf):
    ds = FaceDataset(LIVE_EMBEDDINGS, TRAIN_EMBEDDINGS)
    data, labels, idx_to_name = ds.all()
    clf = clf.fit(data, labels)
    # print(ds.test())
    return clf, idx_to_name


def add_face(clf, num_classes, retrain):
    name = testname
    if not args["train"]:
        name = input("We don't recognize you! Please enter your name:\n").strip().lower()
    increment = 1
    if name == "skip":
        return retrain_classifier(clf), 0
    samples = capture_faces()
    while len(samples) < 50:
        print("We could not capture sufficient samples. Please try again.\n")
        samples = capture_faces()
    embeddings = preprocess_batch(samples)
    embeddings = openFace(embeddings)
    embeddings = embeddings.detach().numpy()

    if name in name_to_idx and retrain:
        print("Face exists, append to embeddings!\n")
        embeddings = update_embedding(LIVE_EMBEDDINGS, embeddings, name)
        increment = 0
    # save name and embeddings
    np.save(LIVE_EMBEDDINGS + "/{}.npy".format(name), embeddings)
    return retrain_classifier(clf), increment

def update_embedding(LIVE_EMBEDDINGS, embeddings, name):
    existing_face = np.load(LIVE_EMBEDDINGS + "/{}.npy".format(name))
    # size is like samplesx128 so change the sample size
    embeddings = np.vstack([existing_face, embeddings])
    np.random.shuffle(embeddings)
    print(len(embeddings))
    return embeddings[:200]

def load_model():
    # TODO: in the future we should look at model persistence to disk
    clf = svm.SVC(kernel="rbf", C=1.0, probability=True)
    #network = BinaryFaceNetwork(device)
    #network.load_state_dict(torch.load("data/binary_face_classifier.pt", map_location=device))
    #clf = BinaryFaceClassifier(network, 0.5)
    ds = FaceDataset(LIVE_EMBEDDINGS, TRAIN_EMBEDDINGS)
    data, labels, idx_to_name = ds.all()
    num_classes = len(np.unique(labels))
    clf = clf.fit(data, labels)
    return clf, num_classes, ds.ix_to_name

def recognize(clf, num_classes, idx_to_name, testing, retrain, hide):
    # to store previous confidences to determine whether a face exists
    prev_conf = deque(maxlen=CONF_TO_STORE)
    f_count = 1
    global CONF_THRESHOLD 
    names = []
    if testing:
        print("Starting the test...")
        # video_capture = cv2.VideoCapture("test_faces/test_videos/lily_singh1.mov")
        # frame_array = []
        test_results = []
    else:
        print("Starting video capture...")
        # video_capture = cv2.VideoCapture(0)
    CONF_THRESHOLD = 0.6 + 0.2/num_classes
    print(CONF_THRESHOLD)
    while True and f_count<= 200:
        # ret is error code but we don't care about it
        ret, frame = video_capture.read()
        if ret:
            # extract and align faces
            if testing: f_count += 1
            rects = extract_faces(frame)
            # print(frame)
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
                unknown_class_prob = probs[0][-1]
                # print(probs)
                predictions = np.argmax(probs, axis=1)
                probs = np.max(probs, axis=1)
                names = [idx_to_name[idx] for idx in predictions]
                # replace all faces below confidence w unknown
                names = [names[i] if probs[i] > CONF_THRESHOLD else "unknown_class" for i in range(len(probs))]
                # print("Hi {}!".format(names))
                for i in range(len(names)):
                    x, y, w, h = face_utils.rect_to_bb(rects[i])
                    cv2.putText(frame, names[i], (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                    if testing: test_results.append(names[i]) 
                # determine if we need to trigger retraining
                # we only retrain if there is one person in the frame and they are unrecognized or there are 0 classes
                if len(faces) == 1:
                    prev_conf.append(unknown_class_prob)
                    if np.mean(prev_conf) > CONF_THRESHOLD and len(prev_conf) == CONF_TO_STORE:
                        (clf, idx_to_name), inc = add_face(clf, num_classes, retrain)
                        print(idx_to_name)
                        num_classes += inc
                        prev_conf.clear()
            else:
                print("No faces detected.")
            if not hide:
                cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('r'):
                (clf, idx_to_name), inc = add_face(clf, num_classes, retrain)
                print(idx_to_name)
                num_classes += inc
                prev_conf.clear()
        else:
            if testing: break
            print("ERROR: no frame captured")
    if testing:  tmod.save_accuracies(testname, test_results)
    video_capture.release()
    cv2.destroyAllWindows()

def write_log(test_res, videoname):
    print("Writing test result logs")
    file1 = open("data/test/test_results/" + testname + " test results.txt", "a+")  # append mode
    file1.write('Test results for ' + file + datetime.datetime.now().strftime(" on %Y-%m-%d %H:%M:%S") +"\n")
    # print('\n'.join(test_res))
    # file1.write('\n'.join(test_res))
    file1.write("Accuracy: \n")
    for name in set(test_res):
        file1.write("   "+name + " {}%\n".format(test_res.count(name)/len(test_res)*100))
    file1.write("Number of frames: {}\n".format(len(test_res)))
    if type(args["note"]) == str:
        file1.write("Note: "+args["note"]+"\n\n") 
    test_res_l = max(len(test_res), 1)
    print("Accuracy is {}%".format(test_res.count(testname)/test_res_l*100))
    # plot_acc(testname,test_res)
    file1.close()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
    except:
        parser.print_help()
        sys.exit(0)    
    parser.add_argument("--gpu", action="store_true", help="run with this flag to run on a GPU")
    parser.add_argument("--train", action="store_true", help="run with this flag to train and dev the model") 
    parser.add_argument("--test", action="store_true", help="run with this flag to test the model") 
    parser.add_argument("--live", action="store_true", help="run with this flag to have a camera demo") 
    parser.add_argument("--retrain", action="store_true", help="run with this flag to retrain for false negatives") 
    parser.add_argument("--plot", action="store_true", help="run with this flag to plot the results") 
    parser.add_argument("--hide", action="store_true", help="run with this flag to hide windows") 
    parser.add_argument('--note', action='store', type=str, help='The text to parse.')
    parser.add_argument('--clean', action='store_true', help='run with this flag to remove previous embeddings')

    args = vars(parser.parse_args())
    device = torch.device("cuda") if args["gpu"] and torch.cuda.is_available() else torch.device("cpu")
    print("Using device {}".format(device))
    openFace = load_openface(device) 

    if args["clean"]:
    
        files = glob.glob(LIVE_EMBEDDINGS + '/*.npy')
        print(files)
        for f in files[:-1]:
            os.remove(f)
        print(glob.glob('data/embeddings/live/*.npy'))

    clf, num_classes, idx_to_name = load_model()
    print(idx_to_name)
    # cannot function as a classifier if less than 2 classes
    assert num_classes >= 2


    if args["train"] or  args["test"]:
        files = glob.glob("data/test/test_videos/*/*")
        random.shuffle(files)
        start = 0
        # files.sort()
        trains = [x for x in files if "train" in x ]
        devs = [x for x in files if "dev" in x ]
        tests = [x for x in files if "testing" in x ]
        files = []
        mode = ""
        if args["train"]:
            files += trains + devs
            mode += "Training "
        if args["test"]:
            files += tests
            mode += "Testing "
        random.shuffle(trains)
        for i in tqdm(range(start, len(files)), total=len(files)):
            file = files[i]
            video_capture = cv2.VideoCapture(file)
            testname = file.split("/")[-2] #.replace("_", " ")
            print(mode + testname + " for file " + file)
            name_to_idx = {idx_to_name[idx]: idx for idx in idx_to_name}
            recognize(clf, num_classes, idx_to_name, True, args["retrain"],  args["hide"])
            clf, num_classes, idx_to_name = load_model()
        print("DONE!! Mode was " + mode)

    if args["live"]:
        print("Starting video capture...")
        video_capture = cv2.VideoCapture(0)
        recognize(clf, num_classes, idx_to_name, args["test"])

    if args["plot"]:
        tmod.plot()
