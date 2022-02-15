import numpy as np
from matplotlib import pyplot as plt


def create_rating_hist(ratings_selected):
    data = ratings_selected['rating']
    data = np.array(data)

    d = np.diff(np.unique(data)).min()
    left_of_first_bin = data.min() - float(d) / 2
    right_of_last_bin = data.max() + float(d) / 2
    n, bins, patches = plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d), edgecolor='white')
    patches[0].set_facecolor('#3ad1f6')
    patches[1].set_facecolor('#3aa2f6')
    patches[2].set_facecolor('#3a6af6')
    patches[3].set_facecolor('#3a3df6')
    patches[4].set_facecolor('#090a89')
    plt.title("Distribuzione valutazioni")
    plt.xlabel("Rating")
    plt.ylabel("Numero valutazioni")
    plt.show()