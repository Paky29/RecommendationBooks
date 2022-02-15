import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import xticks, yticks


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

