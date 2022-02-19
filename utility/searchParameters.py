import itertools
import os
import numpy as np
import pandas as pd
from lightfm import LightFM, cross_validation
from lightfm.evaluation import auc_score, precision_at_k
from model.model import define_features, define_interaction_table, evaluate_model


def sample_hyperparameters():
    while True:
        yield {
            "no_components": np.random.randint(16, 64),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": "warp",
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 15),
            "num_epochs": np.random.randint(5, 50),
            "random_state": 2022
        }


def random_search(train, test, train_weight, item_f, user_f, num_samples=20, num_threads=1):
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(train, epochs=num_epochs, num_threads=num_threads, sample_weight=train_weight,
                  item_features=item_f, user_features=user_f)

        score = auc_score(model, test, item_features=item_f, train_interactions=train,
                          num_threads=num_threads, user_features=user_f).mean()

        hyperparams["num_epochs"] = num_epochs

        yield (score, hyperparams, model)


if __name__ == "__main__":
    # users, books, ratings = pre_process(25, 300)
    datapath = os.path.join("dataset\\dataset_processati", "")
    users = pd.read_csv(datapath + "UsersProcessati.csv")
    books = pd.read_csv(datapath + "BooksProcessati.csv")
    ratings = pd.read_csv(datapath + "RatingsProcessati.csv")

    books_selected = books.merge(ratings, on="isbn")
    books_selected = books_selected[['isbn', 'author', 'pub', 'average_rating', 'category']]
    books_selected = books_selected.drop_duplicates('isbn')

    user_f, item_f, interactions, weights = define_features(users, books_selected, ratings)
    interactions_table = define_interaction_table(ratings)

    train, test = cross_validation.random_train_test_split(interactions, test_percentage=0.30,
                                                           random_state=np.random.RandomState(seed=1))
    train_weights, test_weights = cross_validation.random_train_test_split(weights, test_percentage=0.30,
                                                                           random_state=np.random.RandomState(seed=1))

    (score, hyperparams, model) = max(random_search(train, test, train_weights, item_f, user_f, num_threads=16),
                                      key=lambda x: x[0])

    prec, auc = evaluate_model(model, train, test, item_f, user_f)
    print(prec)
    print(auc)
    print("{}".format(hyperparams))
