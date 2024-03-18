from pyod.models.lunar import LUNAR
from .detector import *
#lunar: https://ojs.aaai.org/index.php/AAAI/article/view/20629
#LUNAR DEVE ESSERE ADDESTRATO SOLO SU ISTANZE BENIGNE: Si aspetta che le y del fit siano tutte 0. 
#Infatti ha solo il metodo decision_function(X) quindi calcola lo score solo sulle istanze di testing poichè quelle di training già sa che sono tutte 0

def lunar(**kwargs):
    class LuNAR(Detector):
        def __init__(self,model_type):
            self.model = LUNAR(model_type = model_type)

        def compile(self, **kwargs):
            pass

        def fit(self, X, y=None, **kwargs):
            self.model.fit(X)

        def predict(self, X, **kwargs):
            pass

        def score(self, X, **kwargs):
            return self.model.decision_function(X)
            

        def print_summary(self):
            print(self.model.get_params(deep=False))
            
    
    model_type = kwargs.get('model_type','WEIGHT')
    return LuNAR(model_type)