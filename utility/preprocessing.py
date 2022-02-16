import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

datapath_proc = os.path.join("dataset\\dataset_processati", "")
datapath_orig = os.path.join("dataset\\dataset_originali", "")


def write_csv(df, percorso):
    df.to_csv(percorso, index=False)


def books_with_ratings(books, ratings):
    books_selected = books.merge(ratings, on="isbn")
    books_selected = books_selected[['isbn', 'author', 'pub', 'average_rating', 'category']]
    books_selected = books_selected.drop_duplicates('isbn')
    return books_selected


def create_gender():
    users = pd.read_csv(
        datapath_orig + "Users.csv",
        usecols=['User-ID', 'Age'],
        dtype={'User-ID': 'str', 'Age': 'float64'})

    # Creazione casuale genere degli utenti
    p = (0.50, 0.50)
    gender = ("M", "F")
    users['gender'] = np.random.choice(gender, size=len(users.index), p=p)
    write_csv(users, datapath_proc + "UsersWithGender.csv")


def pre_process(book_threshold, rating_threshold):
    # Lettura del dataset di libri
    books = pd.read_csv(
        datapath_orig + "Books.csv",
        usecols=['ISBN', 'Book-Title', 'Book-Author', 'Publisher'],
        dtype={'ISBN': 'str', 'Book-Title': 'str', 'Book-Author': 'str', 'Publisher': 'str'})

    # Lettura del dataset categorie
    categories = pd.read_csv(
        datapath_orig + "Preprocessed_data.csv",
        usecols=['isbn', 'Category'],
        dtype={'isbn': 'str', 'Category': 'str'})

    # Lettura del dataset di valutazioni
    ratings = pd.read_csv(
        datapath_orig + "Ratings.csv",
        usecols=['User-ID', 'ISBN', 'Book-Rating'],
        dtype={'User-ID': 'str', 'ISBN': 'str', 'Book-Rating': 'int32'})

    # Lettura del dataset di utenti
    users = pd.read_csv(
        datapath_proc + 'UsersWithGender.csv',
        usecols=['User-ID', 'Age', 'gender'],
        dtype={'User-ID': 'str', 'Age': 'float64', 'gender': 'str'})

    # Modifica nome colonne per facilitarne l'utilizzo
    users.rename(columns={'User-ID': 'user_id', 'Age': 'age'}, inplace=True)
    books = books.rename(columns={'ISBN': 'isbn', 'Book-Title': 'title', 'Book-Author': 'author', 'Publisher': 'pub'})
    categories.rename(columns={'Category': 'category'}, inplace=True)
    ratings.rename(columns={'User-ID': 'user_id', 'ISBN': 'isbn', 'Book-Rating': 'rating'}, inplace=True)

    # Gestione valori mancanti
    books['title'].replace(np.nan, "unknown", inplace=True)
    books['author'].replace(np.nan, 'unknown', inplace=True)
    books['pub'].replace(np.nan, 'unknown', inplace=True)
    categories['category'].replace('9', 'unknown', inplace=True)
    categories['category'].replace(np.nan, 'unknown', inplace=True)

    # Creazione dataframe con numero di rating per libro
    df_books_count = pd.DataFrame(
        ratings.groupby('isbn').size(),
        columns=['count'])

    # Lista dei libri con più di book_threshold valutazioni
    popular_books = list(set(df_books_count.query('count >= @book_threshold').index))
    myfilter = books.isbn.isin(popular_books).values

    # Nuovo dataframe di libri con solo libri con più di booj_threshold valutazioni
    books = books[myfilter]

    # Eliminazione libri duplicati all'interno del dataset
    categories.drop_duplicates(['isbn'], inplace=True)

    # Eliminazioni valutazioni pari a 0
    ratings_selected = ratings.loc[ratings['rating'] >= 1, ['user_id', 'isbn', 'rating']]

    # Selezione utenti con età compresa tra 17 e 50 anni
    users = users.loc[
        ((users['age'] >= 17) & (users['age'] <= 50)) | (np.isnan(users['age'])), ['user_id', 'age', 'gender']]

    # Sostituisco le età con valore NaN con l'età media del dataset
    eta_media = round(users['age'].mean(), 0)
    users['age'].replace(np.nan, eta_media, inplace=True)

    # Scaling rating in una scala da 1 a 5
    scaler = MinMaxScaler(feature_range=(1, 5))
    ratings_selected[["rating"]] = scaler.fit_transform(ratings_selected[["rating"]]).round(0)

    # Selezione delle valutazioni effettuate da utenti più attivi
    x = ratings_selected['user_id'].value_counts() > rating_threshold
    y = x[x].index
    ratings_selected = ratings_selected[ratings_selected['user_id'].isin(y)]

    # Calcolo rating medio
    book_ratings = ratings_selected.groupby(['isbn'])['rating'].mean().reset_index()
    book_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)

    # Merge del dataframe dei libri con quello delle categorie
    book_with_categories = pd.merge(books, categories, on="isbn")

    # Merge del dataframe ottenuto con le valutazioni medie
    book_with_categories = pd.merge(book_with_categories, book_ratings, on="isbn")

    # Merge per determinare i rating relativi a libri validi
    ratings_with_books = pd.merge(ratings_selected, books, on='isbn')
    ratings_selected = ratings_with_books[['user_id', 'isbn', 'rating']]

    # Merge del dataframe degli utenti con quello dei rating selezionati
    users_ratings = users.merge(ratings_selected, on='user_id')
    users = users_ratings[['user_id', 'age', 'gender']]
    users = users.drop_duplicates("user_id")
    ratings_selected = users_ratings[['user_id', 'isbn', 'rating']]

    # salvo i dataframe ottenuti in un file csv
    #write_csv(users, datapath_proc + "UsersProcessati1.csv")
    #write_csv(book_with_categories, datapath_proc + "BooksProcessati1.csv")
    #write_csv(ratings_selected, datapath_proc + "RatingsProcessati1.csv")

    # book_with_categories conterrà anche i libri non coinvolti in nessuna interazione
    return users, book_with_categories, ratings_selected
