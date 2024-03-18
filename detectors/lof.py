from sklearn.neighbors import LocalOutlierFactor

from .detector import *


def lof(**kwargs):
    class LOF(Detector):
        def __init__(self, novelty):
            self.model = LocalOutlierFactor(novelty=novelty)

        def compile(self, **kwargs):
            self.model.compile(optimizer=optimizer, loss=loss)

        def fit(self, X, y, **kwargs):
            self.model.fit(X, y)

        def predict(self, X, **kwargs):
            pass

        def score(self, X, **kwargs):
            scores = self.model.score_samples(X)
            return -scores

        def _score(self, X, y):
            pass

        def print_summary(self):
            print(self.model.get_params(deep=False))

    novelty = kwargs.get('novelty', True)

    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'mse')

    return LOF(novelty)
