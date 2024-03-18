from pyod.models.suod import SUOD
from .detector import *


def suod(**kwargs):
    class Suod(Detector):
        def __init__(self, contamination):
            self.model = SUOD(contamination = contamination)


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

    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'mse')

    return Suod(contamination)