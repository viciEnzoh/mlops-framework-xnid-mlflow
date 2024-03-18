from sklearn.ensemble import RandomForestClassifier

from .classifier import *


def randomforest(**kwargs):
    class RandomForest(Classifier):
        def __init__(self,
                     n_estimators = 100,
                     criterion = 'gini',
                     max_depth = None,
                     min_samples_split = 2,
                     min_samples_leaf = 1,
                     min_weight_fraction_leaf = 0,
                     max_features = 'sqrt',
                     max_leaf_nodes = None,
                     n_jobs = None,
                     random_state = 0):

            self.model = RandomForestClassifier(n_estimators=n_estimators,
                                                criterion = criterion,
                                                max_depth = max_depth,
                                                min_samples_split = min_samples_split,
                                                min_samples_leaf = min_samples_leaf,
                                                min_weight_fraction_leaf = min_weight_fraction_leaf,
                                                max_features = max_features,
                                                max_leaf_nodes = max_leaf_nodes,
                                                n_jobs = n_jobs,
                                                random_state = random_state)

        def compile(self, **kwargs):
            pass

        def fit(self, X, y, **kwargs):
            self.model.fit(X, y)

        def predict(self, X, **kwargs):
            return self.model.predict(X, **kwargs)
        
        def predict_proba(self, X, **kwargs):
            return self.model.predict_proba(X, **kwargs)

        def _score(self, X, y):
            pass

        def print_summary(self):
            print(self.model.get_params(deep=False))

    n_estimators = kwargs.get('n_estimators', 100)
    criterion = kwargs.get('criterion', 'gini')
    max_depth = kwargs.get('max_depth', None)
    min_samples_split = kwargs.get('min_samples_split', 2)
    min_samples_leaf = kwargs.get('min_samples_leaf', 1)
    min_weight_fraction_leaf = kwargs.get('min_weight_fraction_leaf', 0)
    max_features = kwargs.get('max_features', 'sqrt')
    max_leaf_nodes = kwargs.get('leaf_nodes', None)
    n_jobs = kwargs.get('n_jobs', None)
    
    return RandomForest(n_estimators=n_estimators,
                        criterion = criterion,
                        max_depth = max_depth,
                        min_samples_split = min_samples_split,
                        min_samples_leaf = min_samples_leaf,
                        min_weight_fraction_leaf = min_weight_fraction_leaf,
                        max_features = max_features,
                        max_leaf_nodes = max_leaf_nodes,
                        n_jobs = n_jobs)
