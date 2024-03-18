from sklearn.metrics import mean_squared_error as mse
import numpy as np
# from utils import get_xp

# Loading NumPy or CuPy if available
# np = get_xp()


class Detector():
    def __init__(self):
        self.model = None

    def compile(self, **kwargs):
        self.model.compile(**kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def score(self, X, **kwargs):
        X_pred = self.predict(X, **kwargs)
        return self._score(X, X_pred)

    def _score(self, X, y):
        return np.array([mse(t, p, squared=False) for t, p in zip(X, y)])

    def print_summary(self):
        print(self.model.summary())
