from sklearn.model_selection import StratifiedKFold
import numpy as np
from utils import unpack_algorithm
from classifiers.decisiontree import decisiontree
from steps.evaluate import evaluate

def kfold(X, y, model_name, num_folds, configs, task, le, random_state):

    print("[DEBUG] Performing K-fold cross validation (K = " + str(num_folds) + ")")

    performance_collection = {}
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

    #STEPS to iterate through
    #SPLIT (it has to be iterated)
    for c in configs:

        print("[DEBUG] Configuration: " + str(c))

        performance_collection[c] = {}
        performance_collection[c]['accuracy'] = []
        performance_collection[c]['macro_avg_precision'] = []
        performance_collection[c]['macro_avg_recall'] = []
        performance_collection[c]['macro_avg_f1-score'] = []
        performance_collection[c]['weighted_avg_precision'] = []
        performance_collection[c]['weighted_avg_recall'] = []
        performance_collection[c]['weighted_avg_f1-score'] = []
        performance_collection[c]['AUC'] = []
        performance_collection[c]['pAUC'] = []

        n_classes = 0
        n_features = 0
        model = unpack_algorithm(model_name, c, n_features, n_classes, random_state)

        for f in range(num_folds):

            print("[DEBUG] Fold number " + str(f) + "...")

            train_index, test_index = list(skf.split(X, y))[f]
            Xtrain, Xvalid, ytrain, yvalid = X[train_index], X[test_index], y[train_index], y[test_index]

            #TRAIN (fit on chosen model)
            print("[DEBUG] Training...")
            model.fit(Xtrain, ytrain)

            #EVALUATE
            print("[DEBUG] Evaluating...")
            performance, report, ypred = evaluate(model, Xvalid, yvalid, model_name, le, task, random_state)

            if(task == "bMD" or task == "mMD"):
                performance_collection[c]['accuracy'].append(performance['accuracy'])
                performance_collection[c]['macro_avg_precision'].append(performance['macro_avg_precision'])
                performance_collection[c]['macro_avg_recall'].append(performance['macro_avg_recall'])
                performance_collection[c]['macro_avg_f1-score'].append(performance['macro_avg_f1-score'])
                performance_collection[c]['weighted_avg_precision'].append(performance['weighted_avg_precision'])
                performance_collection[c]['weighted_avg_recall'].append(performance['weighted_avg_recall'])
                performance_collection[c]['weighted_avg_f1-score'].append(performance['weighted_avg_f1-score'])
            
            if(task == "AD"):
                performance_collection[c]['AUC'].append(performance['AUC'])
                performance_collection[c]['pAUC'].append(performance['pAUC'])

    #VOTING the best config for actual training+testing
    best_config = vote(performance_collection, task)

    return best_config

####
####

#Utilities
def vote(performance_collection, task):
    
    avg_performance = {}
    best_config = {}

    #Calculating avg. performance of every config
    for c in performance_collection:
        avg_performance[c] = {}

        if(task == "bMD" or task == "mMD"):
            avg_performance[c]['avg_accuracy'] = np.mean(performance_collection[c]['accuracy'])
            avg_performance[c]['std_accuracy'] = np.std(performance_collection[c]['accuracy'])
            avg_performance[c]['avg_precision'] = np.mean(performance_collection[c]['macro_avg_precision'])
            avg_performance[c]['std_precision'] = np.std(performance_collection[c]['macro_avg_precision'])
            avg_performance[c]['avg_recall'] = np.mean(performance_collection[c]['macro_avg_recall'])
            avg_performance[c]['std_recall'] = np.std(performance_collection[c]['macro_avg_recall'])
            avg_performance[c]['avg_f1-score'] = np.mean(performance_collection[c]['macro_avg_f1-score'])
            avg_performance[c]['std_f1-score'] = np.std(performance_collection[c]['macro_avg_f1-score'])

        if(task == "AD"):
            avg_performance[c]['avg_AUC'] = np.mean(performance_collection[c]['AUC'])
            avg_performance[c]['std_AUC'] = np.std(performance_collection[c]['AUC'])
            avg_performance[c]['avg_pAUC'] = np.mean(performance_collection[c]['pAUC'])
            avg_performance[c]['std_pAUC'] = np.std(performance_collection[c]['pAUC'])

    #Once we have collected avg performance for every configuration
    # we have to define a strategy to elect the best configuration
    #
    #
    #
    #


    return best_config

def evaluate(model, Xvalid, yvalid, model_name, le, task, random_state):

    if(task == "bMD" or task == "AD"): binarize_labels = True
    if(task == "mMD"): binarize_labels = False

    performance, report, ypred = evaluate(model, Xvalid, yvalid,
                                            algorithm = model_name,
                                            no_dup = 0,
                                            binarize_labels = binarize_labels,
                                            explain = 0,
                                            le = le,
                                            random_state = random_state,
                                            kfold_mode = True)
    
    return performance, report, ypred