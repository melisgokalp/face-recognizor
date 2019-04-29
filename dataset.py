import argparse
import glob
import os

import numpy as np
from tqdm import tqdm


class FaceDataset:

    def __init__(self, dir, n=None):
        """
        Creates new dataset
        :param dir: path to directory of saved embeddings
        :param n: number of faces to load from the dataset
        """
        self.paths = glob.glob(os.path.join(dir, "*.npy"))
        n = len(self.paths) if n is None else n
        self.data = [[], [], []]
        self.labels = [[], [], []]
        for ix, path in tqdm(enumerate(self.paths[:n]), total=n):
            e = np.load(path)
            l = len(e)
            self.data[0].append(e[:int(0.8 * l)])
            self.data[1].append(e[int(0.8 * l):int(0.9 * l)])
            self.data[2].append(e[int(0.9 * l):])

            self.labels[0].append(np.full(len(e[:int(0.8 * l)]), ix))
            self.labels[1].append(np.full(len(e[int(0.8 * l):int(0.9 * l)]), ix))
            self.labels[2].append(np.full(len(e[int(0.9 * l):]), ix))

        # yes this is shitty but idk why it wasn't working otherwise
        self.data = np.concatenate(self.data[0]), np.concatenate(self.data[1]), np.concatenate(self.data[2])
        self.labels = np.concatenate(self.labels[0]), np.concatenate(self.labels[1]), np.concatenate(self.labels[2])

    def __len__(self):
        return len(self.labels)

    def train(self):
        return self.data[0], self.labels[0]

    def test(self):
        return self.data[1], self.labels[1]

    def dev(self):
        return self.data[2], self.labels[2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("embeddings", help="path to directory with embeddings")
    args = vars(parser.parse_args())
