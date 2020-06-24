import glob

import numpy as np
import plotly
from sklearn.decomposition import PCA

data_map = {}

embeddings = []
labels = []
ix = 0
n = 10
names = []
for face in glob.glob("data/embeddings/live/*.npy")[:n]:
    samples = np.load(face)
    embeddings.append(samples)
    labels.append(np.full(len(samples), ix))
    ix += 1
    name = ((face.split("/")[-1]).split(".")[0])
    names = names + [name]*len(samples)
embeddings = np.concatenate(embeddings)
labels = np.concatenate(labels)

pca = PCA(n_components=3)
reduced = pca.fit_transform(embeddings)
print(np.cumsum(pca.explained_variance_ratio_))
plot = plotly.graph_objs.Scatter(
    x=reduced[:, 0],
    y=reduced[:, 1],
    mode="markers",
    marker=dict(
        color=labels,  # set color equal to a variable
        colorscale='Rainbow'
    ),
    text = names
)

threeplot = plotly.graph_objs.Scatter3d(
    x=reduced[:, 0],
    y=reduced[:, 1],
    z=reduced[:, 2],
    mode="markers",
    marker=dict(
        color=labels,  # set color equal to a variable
        colorscale='Rainbow',
    ),
    text = names
)

plotly.offline.plot([plot], filename="embeddings.html")
plotly.offline.plot([threeplot], filename="embeddings3d.html")
