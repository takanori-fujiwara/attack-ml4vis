import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import torch
from parametric_dr.tsne_nn import TSNE_NN


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
    dist_changes = attr_sensitivity(x_benign, embedder, search_strength)
    most_sensitive_attr = np.argsort(-dist_changes)[0]
    perturbation = np.zeros_like(x_benign)
    perturbation[:, most_sensitive_attr] += 1

    return budget * perturbation


def attr_orth_sensitivity(comparing_attr_idx,
                          x_benign,
                          embedder,
                          strength=0.1):
    _, n_attrs = X.shape
    emb_x_benign = embedder.transform(x_benign)

    x_adv_base = np.copy(x_benign)
    x_adv_base[:, comparing_attr_idx] += strength
    emb_x_adv_base = embedder.transform(x_adv_base)

    v_base = emb_x_adv_base - emb_x_benign

    cos_sims = np.zeros(n_attrs)
    for i in range(n_attrs):
        x_adv_comp = np.copy(x_benign)
        x_adv_comp[:, i] += strength
        emb_adv_comp = embedder.transform(x_adv_comp)

        v_comp = emb_adv_comp - emb_x_benign
        cos_sims[i] = (v_base @ v_comp.T) / (np.linalg.norm(v_base) *
                                             np.linalg.norm(v_comp))

    return 1 - np.abs(cos_sims)


def plot(
        emb_X,
        y_color,
        emb_x_benign=None,
        emb_x_adv=None,
        budgets=None,
        figsize=(4, 4),
        colors={
            0: '#5BA053',  # green
            1: '#ECC854',  # yellow
            2: '#AF7BA1',  # purple
            3: '#507AA6',  # blue
            4: '#F08E39',  # orange
            5: '#78B7B2',  # teal
            6: '#DF585C',  # red
            7: '#9A7460',  # brown
            8: '#FD9EA9',  # pink
            9: '#BAB0AC',  # gray
            -1: '#FFFFFF'  # white
        },
        cmap='coolwarm',
        show_legend=False,
        show_colorbar=False):

    plt.figure(figsize=figsize)
    for i in np.unique(y_color):
        plt.scatter(emb_X[y_color == i, 0],
                    emb_X[y_color == i, 1],
                    c=colors[i],
                    label=i,
                    edgecolor='#AAAAAA',
                    linewidth=0.5)
    if emb_x_benign is not None:
        plt.scatter(emb_x_benign[:, 0],
                    emb_x_benign[:, 1],
                    c='black',
                    label='benign',
                    marker='P',
                    s=120)
    if emb_x_adv is not None:
        if emb_x_adv.shape[0] == 1:
            plt.scatter(emb_x_adv[:, 0],
                        emb_x_adv[:, 1],
                        c='black',
                        label='adver.',
                        marker='X',
                        s=120)
        else:
            c = np.arange(emb_x_adv.shape[0]) if budgets is None else budgets
            plt.scatter(emb_x_adv[:, 0],
                        emb_x_adv[:, 1],
                        c=c,
                        vmax=np.max(np.abs(c)),
                        vmin=0 if np.all(c >= 0) else -np.max(np.abs(c)),
                        label='adver.',
                        marker='X',
                        cmap=cmap,
                        s=100,
                        edgecolor='#666666',
                        linewidth=0.5)
            if show_colorbar:
                plt.colorbar()
    if show_legend:
        plt.legend()
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout()
    plt.show()


## TSNE_NN does not work well for these two small datasets
settings = {
    # 'wine': {
    #     'dataset': datasets.load_wine(),
    #     'attacking_class_label': 2,
    #     'budget': 5,
    #     'budgets': np.arange(0, 16, 1),
    #     'orth_budgets': np.arange(-4, 6, 2)
    # },
    # 'breast_cancer': {
    #     'dataset': datasets.load_breast_cancer(),
    #     'attacking_class_label': 1,
    #     'budget': 15,
    #     'budgets': np.arange(0, 25, 1),
    #     'orth_budgets': np.arange(-10, 12, 2)
    # },
    'digits': {
        'dataset': datasets.load_digits(),
        'attacking_class_label': 0,
        'budget': 20,
        'budgets': np.arange(0, 40, 1),
        'orth_budgets': np.arange(-4, 12, 2)
    }
}

### 0. Loading data
data = 'digits'
s = settings[data]

dataset = s['dataset']
X = dataset.data
y = dataset.target
feat_names = dataset.feature_names
target_names = dataset.target_names
X = scale(X)

embedder = TSNE_NN(
    torch.device('cuda' if torch.cuda.is_available() else 'cpu'), n_epochs=500)
embedder.fit(X)


# TSNE_NN's fit_val is corresponding to transform.
# But, it has a normalization step, which we don't want to apply.
# So, we make transform method by ourself
def transform(data):
    with torch.no_grad():
        embedder.X = torch.from_numpy(data).float()
        embedder.X = embedder.X.to(embedder.device)
        result = embedder.encoder(embedder.X).detach().cpu().numpy()
        return result


embedder.transform = transform

emb_X = embedder.transform(X)
plot(emb_X, y, show_legend=True)

### 1. Black-box attack (one attribute attack)
attacking_class_label = s['attacking_class_label']
x_benign = X[y == attacking_class_label, :][6][np.newaxis, :]

# with too small strength, sometimes cannot find a good attribute for t-sne
# in that case, increase search_strength
# (this is likely due to the algorithm difference. t-sne only focuses on local neighbors.
# plus, parametric tsne implmentation uses sigmoid activation function, which learns slowly
# but has less linearity when compared with other modern activation functions)
budget = s['budget']
perturbation = one_attr_perturbation(x_benign, embedder, search_strength=0.1)
x_adv = x_benign + budget * perturbation

emb_x_benign = embedder.transform(x_benign)
emb_x_adv = embedder.transform(x_adv)
plot(emb_X, y, emb_x_benign=emb_x_benign, emb_x_adv=emb_x_adv)

### 2. Analysis: Transision by budget
budgets = s['budgets']
emb_x_advs = np.zeros((len(budgets), 2))
for i, budget in enumerate(budgets):
    emb_x_advs[i, :] = embedder.transform(x_benign + budget * perturbation)

cmap = LinearSegmentedColormap.from_list('new_cmap',
                                         plt.cm.BuGn(np.linspace(0.15, 1, 10)))
plot(emb_X, [-1] * emb_X.shape[0],
     emb_x_benign=None,
     emb_x_adv=emb_x_advs,
     budgets=budgets,
     cmap=cmap,
     show_colorbar=True)

### 3. Analysis: Histogram for each cultivar
attr_idx = np.where(perturbation > 0)[1][0]
df = pd.DataFrame(X, columns=feat_names)
df['label'] = y
for i, target_name in enumerate(target_names):
    df['label'].iloc[df['label'] == i] = target_name
plotting_data = pd.melt(df, df.columns[-1], df.columns[attr_idx])
g = sns.FacetGrid(plotting_data,
                  col='variable',
                  hue='label',
                  palette=sns.color_palette([
                      '#5BA053', '#ECC854', '#AF7BA1', '#507AA6', '#F08E39',
                      '#78B7B2', '#DF585C', '#9A7460', '#FD9EA9', '#BAB0AC'
                  ]))
g.map(sns.histplot,
      'value',
      linewidth=0,
      binrange=(X[:, attr_idx].min(), X[:, attr_idx].max()))
g.add_legend()
plt.show()

### 4. Analysis: Attribute influencing on the orthogonal direction
orth_attr_idx = np.argmax(
    attr_orth_sensitivity(attr_idx, x_benign, embedder, strength=5))
orth_perturbation = np.zeros_like(perturbation)
orth_perturbation[:, orth_attr_idx] += 1

orth_budgets = s['orth_budgets']
emb_x_advs = np.zeros((len(budgets) * len(orth_budgets), 2))
for i, budget in enumerate(budgets):
    for j, orth_budget in enumerate(orth_budgets):
        emb_x_advs[i * len(orth_budgets) +
                   j, :] = embedder.transform(x_benign +
                                              budget * perturbation +
                                              orth_budget * orth_perturbation)

c = []
for i, budget in enumerate(budgets):
    for j, orth_budget in enumerate(orth_budgets):
        c.append(orth_budget)
c = np.array(c)
plot(emb_X, [-1] * emb_X.shape[0],
     emb_x_benign=None,
     emb_x_adv=emb_x_advs,
     budgets=c,
     show_colorbar=True)
