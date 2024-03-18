from xgboost import XGBClassifier

from .classifier import *


def xgb(**kwargs):
    class XGB(Classifier):
        def __init__(self,
                    n_estimators=100,
                    max_depth = None,
                    learning_rate = None,
                    min_child_weight = None,
                    gamma = None,
                    subsample = None,
                    colsample_bytree = None,
                    reg_alpha = None,
                    random_state = 0):

            self.model = XGBClassifier(use_label_encoder=False,         #LE removed from current version
                                    eval_metric='mlogloss',
                                    n_estimators=n_estimators,
                                    max_depth = max_depth,
                                    learning_rate = learning_rate,
                                    min_child_weight = min_child_weight,
                                    gamma = gamma,
                                    subsample = subsample,
                                    colsample_bytree = colsample_bytree,
                                    reg_alpha = reg_alpha,
                                    random_state = random_state)

        def compile(self, **kwargs):
            pass

        def fit(self, X, y, **kwargs):
            #we have to encode y in order to fit properly the xgb classifier...
            # ... and after we have done that, we have to reset the original encoding
            #so, xgboost has to EMPLOY someway the INTERNAL LABEL ENCODER...
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
    max_depth = kwargs.get('max_depth', None)
    learning_rate = kwargs.get('learning_rate', None)
    min_child_weight = kwargs.get('min_child_weight', None)
    gamma = kwargs.get('gamma', None)
    subsample = kwargs.get('subsample', None)
    colsample_bytree = kwargs.get('colsample_bytree', None)
    reg_alpha = kwargs.get('reg_alpha', None)

    return XGB(n_estimators=n_estimators,
                max_depth = max_depth,
                learning_rate = learning_rate,
                min_child_weight = min_child_weight,
                gamma = gamma,
                subsample = subsample,
                colsample_bytree = colsample_bytree,
                reg_alpha = reg_alpha)
