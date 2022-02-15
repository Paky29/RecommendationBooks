import matplotlib.pyplot as plt
import numpy as np

from lightfm import  cross_validation
from lightfm.evaluation import auc_score, precision_at_k

from model.model import define_features, define_interaction_table, create_model
from utility.preprocessing import books_with_ratings, pre_process

if __name__ == '__main__':
    warp_auc = []
    warp_prec = []
    warp_auc1 = []
    warp_prec1 = []
    x = []

    #create_gender()

    for i in range(0,501,50):
        x.append(i)
        users, books, ratings=pre_process(40, i)
        books_selected = books_with_ratings(books, ratings)

        user_f, item_f, interactions, weights = define_features(users, books_selected, ratings)
        interactions_table = define_interaction_table(ratings)

        train, test = cross_validation.random_train_test_split(interactions, test_percentage=0.30,
                                                           random_state=np.random.RandomState(seed=1))
        train_weights, test_weights = cross_validation.random_train_test_split(weights, test_percentage=0.30,
                                                                           random_state=np.random.RandomState(seed=1))

        warp_model = create_model(train, train_weights,user_f,item_f)

        warp_auc.append(auc_score(warp_model, test, train_interactions=train, item_features=item_f, user_features=user_f).mean())
        warp_auc1.append(auc_score(warp_model, train, item_features=item_f, user_features=user_f).mean())

        warp_prec.append(precision_at_k(warp_model, test, k=5, item_features=item_f, user_features=user_f, train_interactions=train).mean())
        warp_prec1.append(precision_at_k(warp_model, train, k=5, item_features=item_f, user_features=user_f).mean())

    plt.plot(x, np.array(warp_auc))
    plt.plot(x, np.array(warp_auc1))
    plt.plot(x, np.array(warp_prec))
    plt.plot(x, np.array(warp_prec1))
    plt.legend(['AUC TE','AUC TR', 'PREC TE', 'PREC TR'], loc='upper left')
    plt.show()