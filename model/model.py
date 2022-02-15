from turtle import pd

import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM


def feature_colon_value_user(my_list):
    result = []
    ll = ['age', 'gender']
    aa = my_list
    for x, y in zip(ll, aa):
        res = str(x) + ":" + str(y)
        result.append(res)
    return result

def feature_colon_value_item(my_list):
    result = []
    ll = ['author', 'pub', 'average_rating', 'category']
    aa = my_list
    for x, y in zip(ll, aa):
        res = str(x) + ":" + str(y)
        result.append(res)
    return result


def define_features(users, books, ratings):
    # creazione lista di feature dell'utente
    uf = []
    col1 = ['age'] * len(users.age.unique()) + ['gender'] * len(users.gender.unique())
    unique_f1 = list(users.age.unique()) + list(users.gender.unique())
    for x, y in zip(col1, unique_f1):
        res = str(x) + ":" + str(y)
        uf.append(res)

    # creazione lista di feature del libro
    itf = []
    col2 = ['author'] * len(books.author.unique()) + ['pub'] * len(books.pub.unique()) + \
           ['category'] * len(books.category.unique()) + ['average_rating'] * len(books.average_rating.unique())
    unique_f2 = list(books.author.unique()) + list(
        books.pub.unique()) + list(books.category.unique()) + list(
        books.average_rating.unique())
    for x, y in zip(col2, unique_f2):
        res = str(x) + ":" + str(y)
        itf.append(res)

    # creazione dataset e chiamata metodo fit
    dataset = Dataset()
    dataset.fit(
        ratings['user_id'].unique(),
        ratings['isbn'].unique(),
        user_features=uf,
        item_features=itf
    )
    # creazione delle interazioni utente-libro con i relativi pesi
    (interactions, weights) = dataset.build_interactions([(x[0], x[1], x[2]) for x in ratings.values])

    # creazione lista di features per ogni utente
    ad_subset = users[['age', 'gender']]
    ad_list = [list(x) for x in ad_subset.values]
    feature_list1 = []
    for us in ad_list:
        feature_list1.append(feature_colon_value_user(us))
    user_tuple = list(zip(users.user_id, feature_list1))

    # creazione lista di features per ogni libro
    id_subset = books[['author', 'pub', 'average_rating', 'category']]
    id_list = [list(x) for x in id_subset.values]
    feature_list2 = []
    for item in id_list:
        feature_list2.append(feature_colon_value_item(item))
    item_tuple = list(zip(books.isbn, feature_list2))

    # creazione lista di feature dei libri in formato CSR
    item_features = dataset.build_item_features(item_tuple, normalize=False)

    # creazione lista di feature degli utenti in formato CSR
    user_features = dataset.build_user_features(user_tuple, normalize=False)

    return user_features, item_features, weights, interactions


# creiamo il modello utilizzando le interazioni, i pesi, le features degli utenti e le features dei libri
def create_model(interactions, weights, user_features, item_features):
    # impostiamo i parametri del modello
    model = LightFM(loss='warp',
                    random_state=2022,
                    learning_schedule='adadelta',
                    no_components=52,
                    user_alpha=2.6928223797133117e-09,
                    item_alpha=7.58080022366649e-09,
                    max_sampled=7)

    # chiamiamo il metodo fit del modello
    model.fit(interactions,
              sample_weight=weights,
              user_features=user_features,
              item_features=item_features,
              num_threads=16,
              epochs=35)
    return model