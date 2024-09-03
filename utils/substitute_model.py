import numpy as np
from sklearn.decomposition import PCA

# Note: pytorch does not support Python3.11 yet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


class SubstituteDR():

    def __init__(self,
                 ref_emb=None,
                 n_components=2,
                 encoder=None,
                 learning_rate=1e-3,
                 batch_size=None,
                 loss_fn=F.huber_loss):
        self.ref_emb = ref_emb
        if isinstance(self.ref_emb, np.ndarray):
            self.ref_emb = torch.from_numpy(ref_emb).float()
        self.n_components = n_components
        self.encoder = encoder
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_fn = loss_fn

    def fit(self, X, max_epochs=100):
        if self.batch_size is None:
            self.batch_size = min(X.shape[0], 1000)

        dataset = CustomDataset(torch.Tensor(np.arange(X.shape[0])).long())

        trainer = pl.Trainer(accelerator='auto',
                             devices='auto',
                             max_epochs=max_epochs)

        if self.encoder is None:
            self.encoder = DefaultEncoder(X.shape[1:],
                                          n_components=self.n_components)

        self.model = Model(X=torch.Tensor(X).float(),
                           learning_rate=self.learning_rate,
                           encoder=self.encoder,
                           loss_fn=self.loss_fn,
                           ref_emb=self.ref_emb,
                           device='auto')

        trainer.fit(model=self.model,
                    datamodule=DataModule(dataset, self.batch_size))

    @torch.no_grad()
    def transform(self, X):
        return self.model.encoder(
            torch.Tensor(X).float()).detach().cpu().numpy()


class DataModule(pl.LightningDataModule):

    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        # num_workers should be 0 as we are accessing to one array/tensor
        return DataLoader(dataset=self.dataset,
                          batch_size=self.batch_size,
                          num_workers=0,
                          drop_last=True)


class CustomDataset(Dataset):

    def __init__(self, indices):
        shuffled_indices = np.random.permutation(len(indices))
        self.indices = indices[shuffled_indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.indices[index]


class Model(pl.LightningModule):

    def __init__(self,
                 X,
                 learning_rate,
                 encoder,
                 ref_emb,
                 loss_fn=F.huber_loss,
                 device='auto'):
        super().__init__()

        if device == 'auto':
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        self.X = X
        self.lr = learning_rate
        self.encoder = encoder.to(self._device)
        self.ref_emb = ref_emb
        self.loss_fn = loss_fn

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        indices = batch
        emb = self.encoder(self.X[indices])

        encoder_loss = self.loss_fn(self.ref_emb[indices], emb)

        self.log('loss', encoder_loss, prog_bar=True)

        return encoder_loss


class DefaultEncoder(nn.Module):

    def __init__(self, dims, n_components=3, hidden_size=100):
        super(DefaultEncoder, self).__init__()
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


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import datasets
    from sklearn.preprocessing import scale

    dataset = datasets.load_wine()
    X = dataset.data
    y = dataset.target
    X = scale(X)

    emb_X = np.load('emb_X.npy')
    subs_encoder = SubstituteDR(ref_emb=emb_X)
    subs_encoder.fit(X, max_epochs=500)

    emb_X_subs = subs_encoder.transform(X)

    plt.figure(figsize=(4, 4))
    plt.scatter(emb_X[:, 0], emb_X[:, 1], c=y)
    plt.show()