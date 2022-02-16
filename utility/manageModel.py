import pickle


# salva il modello sul disco
def store_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))


# carica il modello dal disco
def load_model(filename):
    model = pickle.load(open(filename, 'rb'))
    return model
