import math

from .detector import *
from .rapp import *


def kitnet(**kwargs):
    # Kitsune: https://gangw.web.illinois.edu/class/cs598/papers/NDSS18-intrusion.pdf
    class KitNET(Detector):
        def __init__(self, input_size, red_fact, k, train_distance, equalize_ensemble, normalize_ensemble):
            self.k = k
            input_sizes = [input_size // k] * k
            for i in range(input_size % k):
                input_sizes[i] += 1
            self.ensemble = [rapp(input_size=input_size, red_fact=red_fact, epochs=epochs) for input_size in input_sizes]
            self.output = rapp(input_size=k, red_fact=red_fact)
            self.train_distance = train_distance
            self.normalize_ensemble = normalize_ensemble
            self.features_indexes = []
            for i in range(k):
                self.features_indexes.append([])
            if equalize_ensemble:
                for i in range(input_size):
                    self.features_indexes[i % k].append(i)
            else:
                for i in range(k):
                    self.features_indexes[i] = list(
                        range(sum(input_sizes[:i]) if i > 0 else 0, sum(input_sizes[:i + 1])))
            print('Features grouping:', self.features_indexes)
            assert list(sorted([w for v in self.features_indexes for w in v])) == list(range(input_size))

        def compile(self, **kwargs):
            for ensemble in self.ensemble:
                ensemble.compile(**kwargs)
            self.output.compile(**kwargs)

        def fit(self, X, y, **kwargs):
            y_score = np.zeros((X.shape[0], self.k))
            for i, ensemble in enumerate(self.ensemble):
                print('Training ensemble %d/%d...' % (i + 1, self.k))
                ensemble.fit(X[:, self.features_indexes[i]], y[:, self.features_indexes[i]], **kwargs)
                y_score[:, i] = ensemble.score(X[:, self.features_indexes[i]], rapp_mode=self.train_distance)
            if self.normalize_ensemble:
                y_score /= y_score.sum(axis=1).reshape(-1, 1)
            print('Training output...')
            self.output.fit(y_score, y_score, epochs=epochs, **kwargs)

        def score(self, X, **kwargs):
            y_score = np.zeros((X.shape[0], self.k))
            for i, ensemble in enumerate(self.ensemble):
                y_score[:, i] = ensemble.score(X[:, self.features_indexes[i]], rapp_mode=self.train_distance)
            if self.normalize_ensemble:
                y_score /= y_score.sum(axis=1).reshape(-1, 1)
            return self.output.score(y_score, **kwargs)

        def print_summary(self):
            for ensemble in self.ensemble:
                ensemble.print_summary()
            self.output.print_summary()

    input_size = kwargs['input_size']
    red_fact = kwargs.get('red_fact', [.5])
    k = kwargs.get('k', 5)
    train_distance = kwargs.get('train_distance', 'normal')
    equalize_ensemble = kwargs.get('equalize_ensemble', False)
    normalize_ensemble = kwargs.get('normalize_ensemble', False)
    epochs = kwargs.get('epochs', 100)
    
    return KitNET(input_size, red_fact, k, train_distance, equalize_ensemble, normalize_ensemble)
