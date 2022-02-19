import numpy as np
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score


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


# creiamo una pivot table delle interazioni utile per mostrare le raccomandazioni
def define_interaction_table(ratings):
    user_book_interaction = pd.pivot_table(ratings, index='user_id', columns='isbn', values='rating')
    user_book_interaction = user_book_interaction.fillna(0)
    return user_book_interaction


# valuta il modello in base alle metriche di precisione e AUC
def evaluate_model(model, train, test, user_features, item_features):
    train_precision = precision_at_k(model, train, k=5, item_features=item_features, user_features=user_features).mean()
    test_precision = precision_at_k(model, test, k=5, item_features=item_features, user_features=user_features,
                                    train_interactions=train).mean()

    train_auc = auc_score(model, train, item_features=item_features, user_features=user_features).mean()
    test_auc = auc_score(model, test, item_features=item_features, user_features=user_features,
                         train_interactions=train).mean()

    return 'Precision: train %.2f, test %.2f.' % (train_precision, test_precision), 'AUC: train %.2f, test %.2f.' % (
        train_auc, test_auc)


# creiamo il dizionario dei libri
def dizionario_item(item):
    item_dict = {}
    df = item[['isbn', 'title']].sort_values('isbn').reset_index()
    for i in range(df.shape[0]):
        item_dict[(df.loc[i, 'isbn'])] = df.loc[i, 'title']
    return item_dict


# creiamo il dizionario degli utenti
def dizionario_user(ui_interactions):
    user_id = list(ui_interactions.index)
    user_dict = {}
    counter = 0
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_id, user_dict


def recommend_unknown_user(model, interactions, item_dict, user_features, item_features, nrec_items=5):
    # otteniamo numero di utenti e numero di libri delle interazioni
    n_users, n_items = interactions.shape
    # otteniamo i punteggi relativi a un utente sconosciuto per ogni libro,
    # utilizzando le feature dei libri e dell'utente stesso
    scores = pd.Series(model.predict(0, np.arange(n_items), item_features=item_features, user_features=user_features))
    # impostiamo come index dei punteggi le colonne delle interazioni, che sono gli isbn dei libri
    scores.index = interactions.columns
    # ordiniamo i punteggi in ordine decrescente
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    # selezioniamo i primi 5 libri
    return_score_list = scores[0:nrec_items]
    # otteniamo la lista dei titoli relativi ai libri selezionati
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    print("\n Recommended Items:")
    counter = 1
    # stampiamo i titoli dei libri ottenuti
    for i in scores:
        print(str(counter) + '- ' + i)
        counter += 1
    return scores


def recommend_user(model, interactions, user_id, user_map, item_dict, user_features, item_features, threshold=0,
                   nrec_items=5):
    # otteniamo numero di utenti e numero di libri delle interazioni
    n_users, n_items = interactions.shape
    # otteniamo un id dell'utente da usare nella predizione
    user_x = user_map[user_id]
    # otteniamo i punteggi relativi ad un utente per ogni libro, utilizzando anche le feature dei libri
    scores = pd.Series(model.predict(user_x, np.arange(n_items), item_features=item_features,
                                     user_features=user_features))
    # impostiamo come index dei punteggi le colonne delle interazioni, che sono gli isbn dei libri
    scores.index = interactions.columns
    # ordiniamo i punteggi in ordine decrescente
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    # otteniamo i libri già presi in prestito dall'utente
    known_items = list(pd.Series(interactions.loc[user_id, :] \
                                     [interactions.loc[user_id, :] > threshold].index).sort_values(ascending=False))
    # scartiamo dalla lista dei punteggi dei libri quelli già presi in prestito dall'utente
    scores = [x for x in scores if x not in known_items]
    # selezioniamo i primi 5 libri
    return_score_list = scores[0:nrec_items]
    # otteniamo la lista dei titoli relativi ai libri selezionati
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    print_known_item(known_items, user_id, item_dict)
    print("\n Recommended Items:")
    counter = 1
    # stampiamo i titoli dei libri ottenuti
    for i in scores:
        print(str(counter) + '- ' + i)
        counter += 1
    return scores


def print_known_item(known_items, user_id, item_dict):
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    print("User: " + str(user_id))
    print("Known Likes:")
    counter = 1
    for i in known_items:
        print(str(counter) + '- ' + i)
        counter += 1