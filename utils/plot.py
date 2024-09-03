import numpy as np
import matplotlib.pyplot as plt


def emb_plot(
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
        show_colorbar=False,
        s=60,
        marker_s=120):

    plt.figure(figsize=figsize)
    for i in np.unique(y_color):
        plt.scatter(emb_X[y_color == i, 0],
                    emb_X[y_color == i, 1],
                    c=colors[i],
                    label=i,
                    edgecolor='#AAAAAA',
                    linewidth=0.5,
                    s=s)
    if emb_x_benign is not None:
        plt.scatter(emb_x_benign[:, 0],
                    emb_x_benign[:, 1],
                    c='black',
                    label='benign',
                    marker='P',
                    s=marker_s)
    if emb_x_adv is not None:
        if emb_x_adv.shape[0] == 1:
            plt.scatter(emb_x_adv[:, 0],
                        emb_x_adv[:, 1],
                        c='black',
                        label='adver.',
                        marker='X',
                        s=marker_s)
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
                        s=marker_s,
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
