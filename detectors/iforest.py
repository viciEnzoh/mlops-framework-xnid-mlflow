from sklearn.ensemble import IsolationForest

from .detector import *


def iforest(random_state, **kwargs):
    class iForest(Detector):
        def __init__(self, n_estimators, max_samples, contamination, random_state):
            self.model = IsolationForest(
                n_estimators=n_estimators, max_samples=max_samples, contamination=contamination, random_state=random_state)

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

    n_estimators = kwargs.get('n_estimators', 100)
    max_samples = kwargs.get('max_samples', 'auto')
    contamination = kwargs.get('contamination', .2)

    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'mse')

    return iForest(n_estimators, max_samples, contamination, random_state)
