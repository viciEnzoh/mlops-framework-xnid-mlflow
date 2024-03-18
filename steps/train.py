
from typing import Dict, Any
import numpy as np
from utils import unpack_algorithm
from classifiers.classifiers_params import sup_learning as sl, unsup_learning as ul

def train(Xtrain, ytrain, algorithm, params, random_state):

    n_classes = len(np.unique(ytrain))
    n_features = len(Xtrain[0])

    model = unpack_algorithm(algorithm, params, n_features, n_classes, random_state)

    #Training distinguishing between supervised and unsupervised learning
    if(algorithm in sl): model.fit(X=Xtrain, y=ytrain)
    elif(algorithm in ul): model.fit(X=Xtrain, y=Xtrain)
    else: print("[DEBUG] Errore: model not present neither among supervised nor unsupervised algorithms")

    return model
