import copy

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

from utils.plot import emb_plot as plot
from umap.parametric_umap import ParametricUMAP

import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

### Loading and preprocessing data
dataset = datasets.load_wine()
X = dataset.data
y = dataset.target
feat_names = dataset.feature_names
target_names = dataset.target_names
X = scale(X)

### Prepare trained ParametricUMAP model (TensorFlow based)
embedder = ParametricUMAP(encoder=None).fit(X)
emb_X = embedder.transform(X)
plot(emb_X, y, show_legend=False)


### 1. One attribute attack
def attr_sensitivity(x_benign, embedder, strength=0.1):
    _, n_attrs = X.shape
    emb_x_benign = embedder.transform(x_benign)

    dists = np.zeros(n_attrs)
    for i in range(n_attrs):
        x_adv = np.copy(x_benign)
        x_adv[:, i] += strength
        emb_x_adv = embedder.transform(x_adv)

        dists[i] = np.linalg.norm(emb_x_adv - emb_x_benign)

    return dists


def one_attr_perturbation(x_benign, embedder, search_strength=0.1, budget=1):
    dist_changes = attr_sensitivity(x_benign, embedder, strength=search_strength)
    most_sensitive_attr = np.argsort(-dist_changes)[0]
    perturbation = np.zeros_like(x_benign)
    perturbation[:, most_sensitive_attr] += 1

    return budget * perturbation


attacking_class_label = 2
x_benign = X[y == attacking_class_label, :][6][np.newaxis, :]

budget = 10
perturbation = one_attr_perturbation(x_benign, embedder)
x_adv = x_benign + budget * perturbation

emb_x_benign = embedder.transform(x_benign)
emb_x_adv = embedder.transform(x_adv)
plot(emb_X, y, emb_x_benign=emb_x_benign, emb_x_adv=emb_x_adv)

# transient by budgets
budgets = np.arange(0, 16, 1)
emb_x_advs = np.zeros((len(budgets), 2))
for i, budget in enumerate(budgets):
    emb_x_advs[i, :] = embedder.transform(x_benign + budget * perturbation)

cmap = LinearSegmentedColormap.from_list(
    "new_cmap", plt.cm.BuGn(np.linspace(0.15, 1, 10))
)
plot(
    emb_X,
    [-1] * emb_X.shape[0],
    emb_x_benign=None,
    emb_x_adv=emb_x_advs,
    budgets=budgets,
    cmap=cmap,
    show_colorbar=False,
)

attr_idx = np.where(perturbation > 0)[1][0]
df = pd.DataFrame(np.vstack((X, x_adv)), columns=feat_names)
df["label"] = list(y) + ["class_3"]
for i, target_name in enumerate(target_names):
    df["label"].iloc[df["label"] == i] = target_name
plotting_data = pd.melt(df, df.columns[-1], df.columns[attr_idx])
g = sns.FacetGrid(
    plotting_data,
    col="variable",
    hue="label",
    palette=sns.color_palette(["#5BA053", "#ECC854", "#AF7BA1", "#000000"]),
)
g.map(
    sns.histplot,
    "value",
    linewidth=0,
    bins=np.arange(plotting_data.value.min(), plotting_data.value.max() + 0.3, 0.3),
)
g.add_legend()
plt.show()

# Note: torch related modules need to be loaded here to avoid the conflict with PUMAP
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.substitute_model import SubstituteDR


### Prepare substitute model (PyTorch based). Stealing/mimicking the trained model
# define any encoder. This one has smaller size than PUMAP's default
class SubstituteEncoder(nn.Module):

    def __init__(self, dims, n_components=2, hidden_size=50):
        super(SubstituteEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.product(dims), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_components),
        )

    def forward(self, x):
        return self.encoder(x)


subs_embedder = SubstituteDR(ref_emb=emb_X, encoder=SubstituteEncoder(X.shape[1:]))
subs_embedder.fit(X, max_epochs=2000)
emb_X_subs = subs_embedder.transform(X)
plot(emb_X_subs, y)


### Demonstrating attacks using linearity of NNs
class AttrCoordNN(nn.Module):

    def __init__(self, n_attrs, encoder, mask=None, fixed_values=None):
        super(AttrCoordNN, self).__init__()
        self.encoder = copy.deepcopy(encoder)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.attr_value_learner = nn.Sequential(
            nn.Linear(1, n_attrs),
        )

        self.mask = torch.Tensor(np.ones((n_attrs))).int()
        if mask is not None:
            self.mask = torch.Tensor(mask).int()

        self.fixed_values = torch.Tensor(np.zeros((n_attrs))).float()
        if fixed_values is not None:
            self.fixed_values = torch.Tensor(fixed_values).float()

    def forward(self, x):
        return self.encoder(
            self.attr_value_learner(torch.eye(1)) * self.mask
            + self.fixed_values * (1 - self.mask)
        )

    def generate_instance(self):
        return self.attr_value_learner(torch.eye(1)) * self.mask + self.fixed_values * (
            1 - self.mask
        )


## Use all attributes to generate x
# ref_emb must be a 2d array (IMPORTANT)
aiming_pos = np.array([[-2.5, -2.5]])
attr_coord_mapper = SubstituteDR(
    ref_emb=aiming_pos,
    encoder=AttrCoordNN(n_attrs=X.shape[1], encoder=subs_embedder.encoder),
)
attr_coord_mapper.fit(np.zeros((1, X.shape[1])), max_epochs=5000)

inst = attr_coord_mapper.encoder.generate_instance().detach().cpu().numpy()

print(subs_embedder.transform(inst))
print(embedder.transform(inst))

plot(
    emb_X_subs,
    y,
    emb_x_benign=subs_embedder.transform(X[117][None, :]),
    emb_x_adv=subs_embedder.transform(inst),
    marker_s=200,
)
plot(
    emb_X,
    y,
    emb_x_benign=embedder.transform(X[117][None, :]),
    emb_x_adv=embedder.transform(inst),
    marker_s=200,
)

## Use only two attributes to generate x
masks = [
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
]
for mask in masks:
    attr_coord_mapper = SubstituteDR(
        ref_emb=aiming_pos,
        encoder=AttrCoordNN(
            n_attrs=X.shape[1],
            encoder=subs_embedder.encoder,
            mask=mask,
            fixed_values=X[117],
        ),
    )
    attr_coord_mapper.fit(np.zeros((1, X.shape[1])), max_epochs=10000)

    inst = attr_coord_mapper.encoder.generate_instance().detach().cpu().numpy()

    print(subs_embedder.transform(inst))
    print(embedder.transform(inst))

    plot(
        emb_X_subs,
        y,
        emb_x_benign=subs_embedder.transform(X[117][None, :]),
        emb_x_adv=subs_embedder.transform(inst),
        marker_s=200,
    )
    plot(
        emb_X,
        y,
        emb_x_benign=embedder.transform(X[117][None, :]),
        emb_x_adv=embedder.transform(inst),
        marker_s=200,
    )

# ## This code examine all possible patterns
# ## even when we only change two of the attributes
# fixed_values = X[117]
# masks = []
# mse_losses = []
# for i in range(X.shape[1]):
#     for j in range(i + 1, X.shape[1]):
#         mask = np.zeros((X.shape[1]), dtype=int)
#         mask[i] = 1
#         mask[j] = 1
#         masks.append(mask)

#         attr_coord_mapper = SubstituteDR(ref_emb=aiming_pos,
#                                          encoder=AttrCoordNN(
#                                              n_attrs=X.shape[1],
#                                              encoder=subs_embedder.encoder,
#                                              mask=mask,
#                                              fixed_values=fixed_values))
#         attr_coord_mapper.fit(np.zeros((1, X.shape[1])), max_epochs=2000)

#         inst = attr_coord_mapper.encoder.generate_instance().detach().cpu(
#         ).numpy()
#         emb_inst = subs_embedder.transform(inst)
#         print(np.array(feat_names)[np.where(mask)[0]])
#         print(emb_inst)
#         mse_losses.append(np.linalg.norm(emb_inst - aiming_pos))

# # top 5 ones:
# for i in range(5):
#     mask = masks[np.argsort(mse_losses)[i]]
#     print(
#         np.array(feat_names)[np.where(mask)[0]],
#         mse_losses[np.argsort(mse_losses)[i]])

# for i in range(5):
#     mask = masks[np.argsort(mse_losses)[i]]
#     attr_coord_mapper = SubstituteDR(ref_emb=aiming_pos,
#                                      encoder=AttrCoordNN(
#                                          n_attrs=X.shape[1],
#                                          encoder=subs_embedder.encoder,
#                                          mask=mask,
#                                          fixed_values=fixed_values))
#     attr_coord_mapper.fit(np.zeros((1, X.shape[1])), max_epochs=5000)

#     inst = attr_coord_mapper.encoder.generate_instance().detach().cpu().numpy()

#     print(subs_embedder.transform(inst))
#     print(embedder.transform(inst))

#     plot(emb_X_subs,
#          y,
#          emb_x_benign=fixed_values[None, :],
#          emb_x_adv=subs_embedder.transform(inst))
#     plot(emb_X,
#          y,
#          emb_x_benign=fixed_values[None, :],
#          emb_x_adv=embedder.transform(inst))

## Generating a fake cluster
x_benign = X[136]
mask = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
aiming_positions = embedder.transform(
    X[
        np.array(
            [
                59,
                62,
                63,
                66,
                67,
                72,
                74,
                75,
                76,
                79,
                80,
                81,
                84,
                85,
                93,
                94,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                106,
                108,
                109,
                110,
                111,
                114,
                116,
                117,
                119,
                120,
                122,
                123,
                124,
                125,
                126,
                128,
                129,
            ]
        ),
        :,
    ]
)

x_advs = np.zeros((aiming_positions.shape[0], X.shape[1]))
for i, aiming_pos in enumerate(aiming_positions):
    # ref_emb must be a 2d array (IMPORTANT)
    attr_coord_mapper = SubstituteDR(
        ref_emb=aiming_pos[None, :],
        encoder=AttrCoordNN(
            n_attrs=X.shape[1],
            encoder=subs_embedder.encoder,
            mask=mask,
            fixed_values=x_benign,
        ),
    )
    attr_coord_mapper.fit(np.zeros((1, X.shape[1])), max_epochs=5000)
    inst = attr_coord_mapper.encoder.generate_instance().detach().cpu().numpy()
    x_advs[i, :] = inst[0]

plot(
    emb_X_subs,
    y,
    emb_x_adv=embedder.transform(x_advs),
    cmap=LinearSegmentedColormap.from_list(
        "new_cmap", plt.cm.Greys(np.linspace(1, 1, 10))
    ),
)

x_advs_augmented = np.zeros((300, x_advs.shape[1]))
for i in range(x_advs_augmented.shape[0]):
    idx_a = np.random.choice(x_advs.shape[0], 1)
    idx_b = np.random.choice(x_advs.shape[0], 1)
    a_ratio = np.random.rand()
    b_ratio = 1 - a_ratio
    x_advs_augmented[i, :] = x_advs[idx_a] * a_ratio + x_advs[idx_b] * b_ratio

plot(
    emb_X,
    y,
    emb_x_adv=embedder.transform(x_advs_augmented),
    cmap=LinearSegmentedColormap.from_list(
        "new_cmap", plt.cm.Greys(np.linspace(1, 1, 10))
    ),
)

X_with_advs = np.vstack((X, x_advs_augmented))
y_with_advs = np.array(list(y) + [1] * x_advs_augmented.shape[0])
df = pd.DataFrame(X_with_advs, columns=feat_names)
df["label"] = y_with_advs
for i, target_name in enumerate(target_names):
    df["label"].iloc[df["label"] == i] = target_name
plotting_data = pd.melt(df, df.columns[-1], df.columns[attr_idx])
g = sns.FacetGrid(
    plotting_data,
    col="variable",
    hue="label",
    palette=sns.color_palette(
        [
            "#5BA053",
            "#ECC854",
            "#AF7BA1",
            "#507AA6",
            "#F08E39",
            "#78B7B2",
            "#DF585C",
            "#9A7460",
            "#FD9EA9",
            "#BAB0AC",
        ]
    ),
)
g.map(
    sns.histplot,
    "value",
    stat="probability",
    linewidth=0,
    binrange=(X_with_advs[:, attr_idx].min(), X_with_advs[:, attr_idx].max()),
)
g.add_legend()
plt.show()

## Influencing asepct ratio
aiming_pos = np.array([[-70, -70]])
attr_coord_mapper = SubstituteDR(
    ref_emb=aiming_pos,
    encoder=AttrCoordNN(n_attrs=X.shape[1], encoder=subs_embedder.encoder),
    learning_rate=0.1,
)
attr_coord_mapper.fit(np.zeros((1, X.shape[1])), max_epochs=5000)
inst = attr_coord_mapper.encoder.generate_instance().detach().cpu().numpy()
plot(emb_X_subs, y, emb_x_adv=subs_embedder.transform(inst), marker_s=200)
