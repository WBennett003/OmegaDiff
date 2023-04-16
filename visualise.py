import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from utils.tokeniser import Tokeniser
tokeniser = Tokeniser()

def denoising_animation(X):
    fig = plt.figure()
    ims = []
    for i in range(len(X)):
        im = plt.imshow(X[i][0].cpu(), cmap="magma", animated=True, vmin=-4, vmax=4)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return animate

def noising_animation(X):
    pass

def noising_multiplot(X, sections=5):
    pass


def plot_AA_confusion(Y_true, Y_pred, annot=False, get_figure=False):
    
    labels = list(tokeniser.AA.keys())
    n_AAs = len(labels)

    fig, ax = plt.subplots(3, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(8,15))
    
    uniques = np.arange(n_AAs)
    blank = np.zeros(n_AAs)
    True_uniques, True_count = np.unique(Y_true, return_counts=True)
    blank[np.intc(True_uniques)] = True_count    
    True_count = blank

    ax[1].bar(uniques, True_count)
    ax[1].set_xticks(uniques, labels)
    ax[1].set_title("True AA Distrubution")

    blank = np.zeros(n_AAs)
    Pred_uniques, Pred_count = np.unique(Y_pred, return_counts=True)
    blank[np.intc(Pred_uniques)] = Pred_count    
    Pred_count = blank

    ax[2].bar(uniques, Pred_count, label=labels)
    ax[2].set_title("Pred AA Distrubution")
    ax[2].set_xticks(uniques, labels)


    if len(Y_true.shape) == 1:
        Conf_matrix = confusion_matrix(Y_true, Y_pred, labels=uniques)
    else:
        confs = np.zeros((n_AAs, n_AAs))
        for batch in range(Y_true.shape[0]):
            Conf_matrix = confusion_matrix(Y_true[batch], Y_pred[batch], labels=uniques)
            confs += Conf_matrix
        Conf_matrix = confs

    heatmap = sns.heatmap(Conf_matrix, xticklabels=labels, yticklabels=labels, annot=annot, ax=ax[0])
    ax[0].set(xlabel='Pred', ylabel='True')
    fig.tight_layout()
    if get_figure:
        plt.close()
        return fig
    else:
        plt.show()

