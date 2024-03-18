
from steps.ingest import ingest
from steps.split import split
from steps.preprocess import preprocess
from steps.kfold import kfold
from steps.train import train
from steps.evaluate import evaluate

import mlflow
import mlflow.keras

#PIPELINE CLASS
class Pipeline():

    _dataset = ""
    _steps = []
    _algorithm = ""
    _model_name = ""
    _alg_params = {}
    _flavor = ""

    def __init__(self, dataset, steps, algorithm, random_state, params, label,
                 no_dup=0,
                 binarize_labels=0,
                 samples_barrier=300,
                 explain=0,
                 split_proportion=[0.7, 0.3, 0.0],
                 num_folds=10
                 ):

        sklearn_models = ["random-forest", "decision-tree"]
        xgb_models = ["xg-boost"]
        dl_models = ["mlp"]
        #...

        #INITIALIZATION
        self._dataset = dataset
        self._steps = steps
        self._algorithm = algorithm
        self._random_state = random_state
        self._alg_params = params
        self._label = label
        self._no_dup = no_dup
        self._binarize_labels = binarize_labels
        self._samples_barrier = samples_barrier
        self._kfold = kfold
        self._proportion = split_proportion
        self._num_folds = num_folds
        self._features = []
        self._explain = explain

        #PIPELINE ARTIFACTS
        #after INGEST
        self._data = []

        #after SPLIT, PREPROCESS, TRANSFORM
        self._Xtrain = []
        self._Xtest = []
        self._ytrain = []
        self._ytest = []
        self._le = None

        #after KFOLD (if function is invoked)
        self._best_config = {}

        #after EVALUATE
        self._performance = {}
        self._report = {}
        self._ypred = []


        if algorithm in sklearn_models: self._flavor = "sklearn"
        if algorithm in xgb_models: self._flavor = "xgb"
        if algorithm in dl_models: self._flavor = "keras"

    def steps(self):
        print("[DEBUG] Steps: " + str(self._steps))

    def ingest(self):
        self._data, self._features = ingest(self._dataset)
    

    def split(self):
        self._Xtrain, self._Xtest, self._ytrain, self._ytest = split(self._data,
                                                                     self._label,
                                                                     proportion=self._proportion,
                                                                     random_state=self._random_state)
        return len(self._Xtrain), len(self._Xtest)
    
    def preprocess(self):
        self._Xtrain, self._Xtest, self._ytrain, self._ytest, self._le = preprocess(self._Xtrain,
                                                                          self._Xtest,
                                                                          self._ytrain,
                                                                          self._ytest,
                                                                          self._dataset,
                                                                          binarize_labels=self._binarize_labels,
                                                                          samples_barrier=self._samples_barrier)
        
        #returning Label Encoder for logging
        return self._le
    
    def transform():
        pass

    def kfold(self):
        self._best_config = kfold(self._Xtrain, self._ytrain,
                                    self._algorithm,
                                    self._num_folds,
                                    [],        #configs
                                    "bMD",     ###
                                    self._le,
                                    self._random_state)
    
    def train(self):
        #if kfold has been performed, params should be the return value from kfold func, else params as expected as "single-shot" pipeline
        config = self._alg_params if self._best_config == {} else self._best_config
        self._model = train(self._Xtrain, self._ytrain, self._algorithm, params=config, random_state=self._random_state)
        return self._model
    
    def load_model(self, model_name, model_version):
        """
        if(self._flavor == "sklearn"): self._model = mlflow.sklearn.load_model(model_uri = f"models:/{model_name}/{model_version}")
        if(self._flavor == "xgb"): self._model = mlflow.sklearn.load_model(model_uri = f"models:/{model_name}/{model_version}")
        if(self._flavor == "keras"): self._model = mlflow.sklearn.load_model(model_uri = f"models:/{model_name}/{model_version}")
        #..."""
        self._model = mlflow.sklearn.load_model(model_uri = f"models:/{model_name}/{model_version}")
        return self._model

    def evaluate(self):
        self._performance, self._report, self._ypred = evaluate(self._model,
                                                                self._Xtest,
                                                                self._ytest,                                                                
                                                                no_dup=self._no_dup,
                                                                binarize_labels= self._binarize_labels,
                                                                explain=self._explain,
                                                                features=self._features,
                                                                algorithm=self._algorithm,
                                                                Xtrain=self._Xtrain,
                                                                ytrain=self._ytrain,
                                                                le=self._le,
                                                                random_state=self._random_state)
        
        if(not self._binarize_labels): ytest_decoded = self._le.inverse_transform(self._ytest)
        else: ytest_decoded = ['BENIGN' if label == 0 else 'MALWARE' for label in self._ytest]
        return self._performance, self._report, ytest_decoded, self._ypred