# from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from dataset import FaceDataset


workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# For a model pretrained on VGGFace2
# model = InceptionResnetV1(pretrained='vggface2').eval()
# For a model pretrained on CASIA-Webface
# model = InceptionResnetV1(pretrained='casia-webface').eval()

# For an untrained model with 100 classes
# model = InceptionResnetV1(num_classes=100).eval()

# For an untrained 1001-class classifier
# model = InceptionResnetV1(classify=True, num_classes=1001).eval()

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# Define a dataset and data loader
def collate_fn(x):
    return x[0]

def load_model():
    # TODO: in the future we should look at model persistence to disk
    clf = svm.SVC(kernel="linear", gamma='auto', C=1.0, probability=True)
    #network = BinaryFaceNetwork(device)
    #network.load_state_dict(torch.load("data/binary_face_classifier.pt", map_location=device))
    #clf = BinaryFaceClassifier(network, 0.5)
    ds = FaceDataset("data/embeddings/live", "data/embeddings/train")
    data, labels, idx_to_name = ds.all()
    num_classes = len(np.unique(labels))
    clf = clf.fit(data, labels)
    return clf, num_classes, ds.ix_to_name

# dataset = FaceDataset("data/embeddings/live", "data/embeddings/train")
dataset = datasets.ImageFolder('data/dd')

dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
# dataset.idx_to_class = dataset.ix_to_name
# data, labels, idx_to_name = dataset.all()
print(dataset)

loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names))