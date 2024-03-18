from sklearn.tree import DecisionTreeClassifier

from .classifier import *


def decisiontree(**kwargs):
    class DecisionTree(Classifier):
        def __init__(self,
                    criterion = 'gini',
                    splitter = 'best',
                    max_depth = None,
                    min_samples_split = 2,
                    min_samples_leaf = 1,
                    min_weight_fraction_leaf = 0,
                    max_features = None,
                    max_leaf_nodes = None,
                    random_state = 0):
            
            self.model = DecisionTreeClassifier(
                criterion = criterion,
                splitter = splitter,
                max_depth = max_depth,
                min_samples_split = min_samples_split,
                min_samples_leaf = min_samples_leaf,
                min_weight_fraction_leaf = min_weight_fraction_leaf,
                max_features = max_features,
                max_leaf_nodes = max_leaf_nodes,
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

    criterion = kwargs.get('criterion', 'gini')
    splitter = kwargs.get('splitter', 'best')
    max_depth = kwargs.get('max_depth', None)
    min_samples_split = kwargs.get('min_samples_split', 2)
    min_samples_leaf = kwargs.get('min_samples_leaf', 1)
    min_weight_fraction_leaf = kwargs.get('min_weight_fraction_leaf', 0)
    max_features = kwargs.get('max_features', None)
    max_leaf_nodes = kwargs.get('max_leaf_nodes', None)

    return DecisionTree(
                criterion = criterion,
                splitter = splitter,
                max_depth = max_depth,
                min_samples_split = min_samples_split,
                min_samples_leaf = min_samples_leaf,
                min_weight_fraction_leaf = min_weight_fraction_leaf,
                max_features = max_features,
                max_leaf_nodes = max_leaf_nodes)
