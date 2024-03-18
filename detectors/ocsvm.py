from sklearn.svm import OneClassSVM

from .detector import *

def ocsvm(**kwargs):
    #sklearn.svm.OneClassSVM
    class OC_SVM(Detector):
        def __init__(self, kernel, gamma):
            self.model = OneClassSVM(kernel=kernel, gamma=gamma)

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

    kernel = kwargs.get('kernel', 'rbf')
    gamma = kwargs.get('gamma', 'auto')

    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'mse')

    return OC_SVM(kernel, gamma)
