import time

import numpy as np
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k
from matplotlib import pyplot as plt
from matplotlib.pyplot import xticks


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


def create_age_hist(utenti):
    data = utenti['age']
    data = np.array(data)

    d = np.diff(np.unique(data)).min()
    left_of_first_bin = data.min() - float(d) / 2
    right_of_last_bin = data.max() + float(d) / 2
    plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d), edgecolor='white')
    xticks([15, 20, 30, 40, 50])
    plt.title("Distribuzione età")
    plt.xlabel("Età")
    plt.ylabel("Numero utenti")
    plt.show()

def create_age_pie_chart(utenti):
    df_inrange= utenti.loc[((utenti['age'] >= 17) & (utenti['age'] <= 50)), ['user_id', 'age']]
    df_not_inrange = utenti.loc[((utenti['age'] < 17) | (utenti['age'] > 50) | (np.isnan(utenti['age']))), ['user_id', 'age']]
    sizes1=df_inrange['user_id'].count()
    sizes2=df_not_inrange['user_id'].count()
    sizes= [sizes1, sizes2]
    fig1, ax1 = plt.subplots()
    colors = ['#3ad1f6', '#3a6af6']
    ax1.pie(sizes, labels=["17<=age<=50", "other"],
            shadow=True, startangle=90, colors=colors, autopct='%0.01f%%')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle."""
    plt.show()

def create_gender_pie_chart(utenti):
    sizes = (utenti['gender']).value_counts().tolist()

    fig1, ax1 = plt.subplots()
    colors = ['#007FFF', '#FF54A7']
    ax1.pie(sizes, labels=["M", "F"],
            shadow=True, startangle=90, colors=colors, autopct='%0.01f%%')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle."""
    plt.show()

def variation_epochs(train, train_weights, test, item_f, user_f):
    epochs=35
    model = LightFM(loss='warp',
                    random_state=2022,
                    learning_schedule='adadelta',
                    no_components=52,
                    user_alpha=2.6928223797133117e-09,
                    item_alpha=7.58080022366649e-09,
                    max_sampled=7)

    warp_auc = []
    warp_prec = []
    warp_auc1 = []
    warp_prec1 = []
    warp_duration = []

    for epoch in range(epochs+1):
        start = time.time()
        model.fit_partial(train,
              sample_weight=train_weights,
              user_features=user_f,
              item_features=item_f,
              num_threads=16,
              epochs=1)

        warp_duration.append(time.time() - start)
        warp_auc.append(
            auc_score(model, test, train_interactions=train, item_features=item_f, user_features=user_f).mean())
        warp_auc1.append(auc_score(model, train, item_features=item_f, user_features=user_f).mean())

        warp_prec.append(precision_at_k(model, test, k=5, item_features=item_f, user_features=user_f,
                                        train_interactions=train).mean())
        warp_prec1.append(precision_at_k(model, train, k=5, item_features=item_f, user_features=user_f).mean())

    x = np.arange(epochs+1)
    plt.plot(x, np.array(warp_auc))
    plt.plot(x, np.array(warp_auc1))
    plt.plot(x, np.array(warp_prec))
    plt.plot(x, np.array(warp_prec1))
    plt.legend(['AUC TE', 'AUC TR', 'PREC TE', 'PREC TR'], loc='upper left')
    plt.show()

