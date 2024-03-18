from .classifier import *
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def mlp(**kwargs):
    class MLP(Classifier):
        def __init__(self, input_size, output_size, red_fact):
            i = Input(input_size)
            x = i
            for rf in red_fact:
                x = Dense(np.ceil(input_size[1] * rf), activation='relu')(x)
            x = Dropout(.1)(x)
            o = Dense(output_size, activation='softmax')(x)

            self.model = Model(i, o)
            self.model.summary()
            self.model.compile(optimizer=optimizer, loss=loss)

        #gestire early stopping #########

        def fit(self, X, y, **kwargs):
            X = np.array([[xx for xx in x] for x in X])     #necessary for the multiclass detection case
            y = np.array(y)
            super().fit(X, OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1)),
                        epochs=epochs,
                        batch_size=batch_size,
                        **kwargs)

        def predict_proba(self, X, **kwargs):
            X = np.array([[xx for xx in x] for x in X])     #necessary for the multiclass detection case
            return self.model.predict(X)

        def _score(self, X, y):
            pass

    input_size = kwargs['input_size']
    output_size = kwargs['output_size']
    red_fact = kwargs.get('red_fact', [1.5, 1.5])
    
    epochs = kwargs.get('epochs', 2)
    batch_size = kwargs.get('batch_size', 128)

    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'mse')

    return MLP(input_size, output_size, red_fact)
