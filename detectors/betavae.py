from pyod.models.vae import VAE
from .detector import *


def betavae(**kwargs):
    # VAE: https://arxiv.org/abs/1312.6114, https://arxiv.org/pdf/1804.03599.pdf
    class Betavae(Detector):
        def __init__(self,contamination, gamma, capacity, epochs):
            self.model = VAE(contamination=contamination, gamma=gamma, capacity=capacity, epochs=epochs)

        def compile(self, **kwargs):
            self.model.compile(optimizer=optimizer, loss=loss)

        def fit(self, X, y=None, **kwargs):
            self.model.fit(X)

        def predict(self, X, **kwargs):
            pass
        
        def score_training(self, **kwargs):
            return self.model.decision_scores_

        def score(self, X, **kwargs):
            return self.model.decision_function(X)

        def print_summary(self):
            print(self.model.get_params(deep=False))

    contamination = kwargs.get('contamination', 0.1)
    gamma = kwargs.get('gamma',1.0)
    capacity = kwargs.get('capacity', 0.0)
    epochs = kwargs.get('epochs', 100)

    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'mse')

    return Betavae(contamination, gamma, capacity, epochs)