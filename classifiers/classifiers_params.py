classifiers_params_info = {

    #(key -> value) = (param -> default_value)

    #sklearn.tree.DecisionTreeClassifier    
    'decision-tree' : {
        'criterion' : 'gini',
        'splitter' : 'best',
        'max_depth' : None,
        'min_samples_split' : 2,
        'min_samples_leaf' : 1,
        'min_weight_fraction_leaf' : 0,
        'max_features' : None,
        'max_leaf_nodes' : None
    },

    #sklearn.ensemble.RandomForestClassifier
    'random-forest' : {
        'n_estimators' : 100,
        'criterion' : 'gini',
        'max_depth' : None,
        'min_samples_split' : 2,
        'min_samples_leaf' : 1,
        'min_weight_fraction_leaf' : 0,
        'max_features' : 'sqrt',
        'max_leaf_nodes' : None,
        'n_jobs' : None
    },

    #xgboost.XGBClassifier
    'xg-boost' : {
        'n_estimators' : 100,
        'max_depth' : None,
        'learning_rate' : None,
        'min_child_weight' : None,
        'gamma' : None,
        'subsample' : None,
        'colsample_bytree' : None,
        'reg_alpha' : None
    },

    #.classifiers.mlp
    'mlp' : {
        'epochs' : 5,
        'batch_size' : 128,
        'red_fact': [1.5, 1.5],
        'loss' : 'mse',
        'optimizer' : 'adam'
    },

    #.detectors.rapp
    'rapp' : {
        'red_fact' : [.75, .5],
        'rapp_mode' : 'e_rmse',
        'epochs' : 100,
        'optimizer' : 'adam',
        'loss' : 'mse'
    },

    #pyod.models.alad.ALAD
    'alad' : {
        'contamination' : 0.1,
        'optimizer' : 'adam',
        'loss' : 'mse'
    },

    #pyod.models.anogan.ANOGAN
    'anogan' : [],

    #from pyod.models.vae.VAE
    'betavae' : {
        'contamination' : 0.1,
        'gamma' : 1.0,
        'capacity' : 0.0,
        'epochs' : 10,
        'optimizer' : 'adam',
        'loss' : 'mse'
    }
,

    #from pyod.models.ecod.ECOD
    'ecod' : {
        'contamination' : 0.1,
        'optimizer' : 'adam',
        'loss' : 'mse'
    },

    #sklearn.ensemble.IsolationForest
    'iforest': {
        'n_estimators' : 100,
        'max_samples' : 'auto',
        'contamination' : 0.1,
        'optimizer' : 'adam',
        'loss' : 'mse'
    },

    #.detectors.kitnet
    'kitnet' : {
        'red_fact' : [.5],
        'k' : 5,
        'train_distance' : 'normal',
        'equalize_ensemble' : False,
        'normalize_ensemble' : False,
        'epochs' : 100
    },

    #sklearn.neighbors.LocalOutlierFactor
    'lof' : {
        'novelty' : True,
        'optimizer' : 'adam',
        'loss' : 'mse'
    },

    #sklearn.svm.OneClassSVM
    'ocsvm' : [],

    #pyod.models.rod.ROD
    'rod' : [],

    #pyod.models.suod.SUOD
    'suod' : [],

    #pyod.models.vae import VAE
    'vae' : {
        'contamination' : 0.1,
        'gamma' : 1.0,
        'capacity' : 0.0,
        'epochs' : 10,
        'optimizer' : 'adam',
        'loss' : 'mse'
    }

}

eval_list = [
    'n_estimators',
    'max_depth',
    'min_samples_split',
    'min_samples_leaf',
    'max_leaf_nodes',
    'min_weight_fraction_leaf',
    'n_jobs',
    'learning_rate',
    'subsample'
    'epochs',
    'batch_size',
    'novelty',
    'contamination',
    'red_fact'
]

sup_learning = [
    "random-forest",
    "decision-tree",
    "xg-boost",
    "mlp"]

unsup_learning = [
    "rapp",
    "alad",
    "anogan",
    "betavae",
    "ecod",
    "iforest",
    "kitnet",
    "lof",
    "ocsvm",
    "rod",#####
    "suod",#####
    "vae"]