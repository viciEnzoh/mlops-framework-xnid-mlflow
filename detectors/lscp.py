from pyod.models.lscp import LSCP
from .detector import *
from pyod.models.lof import LOF


def lscp(**kwargs):
    class Lscp(Detector):
        def __init__(self,detector_list,contamination): #detector_list DEVE essere almeno di 2 elementi (almeno 2 modelli di PyOD)
            # NOTA : Qualora si volesse utilizzare un altro modello che non sia della libreria PyOD, i modelli devono avere i metodi fit(X) e 
            # decision_function(X)
            self.model = LSCP(detector_list=detector_list, contamination=contamination)
        

        def compile(self, **kwargs):
            pass

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
            
    detector_list = kwargs.get('detector_list',[LOF(),LOF()])
    contamination = kwargs.get('contamination',0.1)
    return Lscp(detector_list,contamination)