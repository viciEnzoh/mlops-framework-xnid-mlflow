import os
import random
from time import process_time
import base64

from classifiers.decisiontree import decisiontree
from classifiers.randomforest import randomforest
from classifiers.xgb import xgb
from classifiers.mlp import mlp
from detectors.rapp import rapp
from detectors.alad import alad
from detectors.anogan import anogan
from detectors.betavae import betavae
from detectors.ecod import ecod
from detectors.iforest import iforest
from detectors.kitnet import kitnet
from detectors.lof import lof
from detectors.lscp import lscp
from detectors.lunar import lunar
from detectors.ocsvm import ocsvm
from detectors.rod import rod
from detectors.suod import suod
from detectors.vae import vae
#from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.callbacks import *
import numpy as np
import pandas
import json

from datasets.extract_statistics import get_statistics

# try:
#     import cupy as cp
# except ImportError:
#     cp = None

# def get_xp(use_cupy_if_available=True):
#     if use_cupy_if_available and cp is not None:
#         return cp
#     return np


class TimeEpochs(Callback):
    """
    Callback used to calculate per-epoch time
    """

    def on_train_begin(self, logs):
        logs = logs or {}
        return

    def on_epoch_begin(self, batch, logs):
        logs = logs or {}
        self.epoch_time_start = process_time()

    def on_epoch_end(self, batch, logs):
        logs = logs or {}
        logs['time'] = (process_time() - self.epoch_time_start)


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
#     try:
#         cp.random.seed(seed)
#     except:
#         print('CuPy not available: using NumPy instead.')
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)  # This is redundant with random, np.random, and tf.random
        tf.config.experimental.enable_op_determinism()
    except:
        pass

def unpack_algorithm(algorithm, params, n_features, n_classes, random_state):

    f = open('models.json')
    models_info = json.load(f)

    print("[DEBUG] Chosen model: " + models_info["algorithm"][algorithm]["extended_name"])

    if(algorithm == "random-forest"):
        print("[DEBUG] Parameters: " + str(params))

        n_estimators = params["n_estimators"]
        criterion = params["criterion"]
        max_depth = params["max_depth"]
        min_samples_split = params["min_samples_split"]
        min_samples_leaf = params["min_samples_leaf"]
        min_weight_fraction_leaf = params["min_weight_fraction_leaf"]
        max_features = params["max_features"]
        max_leaf_nodes = params["max_leaf_nodes"]
        n_jobs = params["n_jobs"]

        model = randomforest(n_estimators=n_estimators,
                            criterion = criterion,
                            max_depth = max_depth,
                            min_samples_split = min_samples_split,
                            min_samples_leaf = min_samples_leaf,
                            min_weight_fraction_leaf = min_weight_fraction_leaf,
                            max_features = max_features,
                            max_leaf_nodes = max_leaf_nodes,
                            n_jobs = n_jobs,
                            random_state = random_state)

    elif(algorithm == "decision-tree"):
        print("[DEBUG] Parameters: " + str(params))

        criterion = params["criterion"]
        splitter = params["splitter"]
        max_depth = params["max_depth"]
        min_samples_split = params["min_samples_split"]
        min_samples_leaf = params["min_samples_leaf"]
        min_weight_fraction_leaf = params["min_weight_fraction_leaf"]
        max_features = params["max_features"]
        max_leaf_nodes = params["max_leaf_nodes"]

        model = decisiontree(criterion = criterion,
                splitter = splitter,
                max_depth = max_depth,
                min_samples_split = min_samples_split,
                min_samples_leaf = min_samples_leaf,
                min_weight_fraction_leaf = min_weight_fraction_leaf,
                max_features = max_features,
                max_leaf_nodes = max_leaf_nodes,
                random_state = random_state)

    elif(algorithm == "xg-boost"):
        print("[DEBUG] Parameters: " + str(params))

        n_estimators = params["n_estimators"]
        max_depth = params["max_depth"]
        learning_rate = params["learning_rate"]
        min_child_weight = params["min_child_weight"]
        gamma = params["gamma"]
        subsample = params["subsample"]
        colsample_bytree = params["colsample_bytree"]
        reg_alpha = params["reg_alpha"]

        model = xgb(n_estimators=n_estimators,
                    max_depth = max_depth,
                    learning_rate = learning_rate,
                    min_child_weight = min_child_weight,
                    gamma = gamma,
                    subsample = subsample,
                    colsample_bytree = colsample_bytree,
                    reg_alpha = reg_alpha,
                    random_state = random_state)

    elif(algorithm == "mlp"):
        print("[DEBUG] Parameters: " + str(params))
        
        epochs = params["epochs"]
        batch_size = params["batch_size"]
        red_fact = params["red_fact"]

        model = mlp(input_size=(None,n_features),
                    output_size=n_classes,
                    red_fact=red_fact,
                    epochs=epochs,
                    batch_size=batch_size)

    elif(algorithm == "rapp"):
        print("[DEBUG] Parameters: " + str(params))

        epochs = params["epochs"]
        red_fact = params["red_fact"]
        rapp_mode = params["rapp_mode"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = rapp(input_size=n_features,
                     red_fact=red_fact,
                     epochs=epochs,
                     rapp_mode=rapp_mode,
                     optimizer = optimizer,
                     loss=loss)

    elif(algorithm == "alad"):
        print("[DEBUG] Parameters: " + str(params))

        contamination = params["contamination"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = alad(contamination = contamination,
                     optimizer = optimizer,
                     loss = loss)

    elif(algorithm == "anogan"):
        print("[DEBUG] Parameters: " + str(params))

        contamination = params["contamination"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = anogan(contamination = contamination,
                     optimizer = optimizer,
                     loss = loss)

    elif(algorithm == "betavae"):
        print("[DEBUG] Parameters: " + str(params))

        contamination = params["contamination"]
        gamma = params["gamma"]
        capacity = params["capacity"]
        epochs = params["epochs"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = betavae(contamination = contamination,
                        gamma = gamma,
                        capacity = capacity,
                        epochs = epochs,
                        optimizer = optimizer,
                        loss = loss)

    elif(algorithm == "ecod"):
        print("[DEBUG] Parameters: " + str(params))

        contamination = params["contamination"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = ecod(contamination = contamination,
                     optimizer = optimizer,
                     loss = loss)

    elif(algorithm == "iforest"):
        print("[DEBUG] Parameters: " + str(params))

        n_estimators = int(params["n_estimators"])
        max_samples = params["max_samples"]
        contamination = params["contamination"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = iforest(n_estimators = n_estimators,
                        max_samples = max_samples,
                        contamination = contamination,
                        optimizer = optimizer,
                        loss = loss,
                        random_state = random_state)

    elif(algorithm == "kitnet"):
        print("[DEBUG] Parameters: " + str(params))

        red_fact = params["red_fact"]
        k = params["k"]
        train_distance = params["train_distance"]
        equalize_ensemble = params["equalize_ensemble"]
        normalize_ensemble = params["normalize_ensemble"]
        epochs = params["epochs"]

        model = kitnet(input_size = n_features,
                        red_fact = red_fact,
                        k = k,
                        train_distance = train_distance,
                        equalize_ensemble = equalize_ensemble,
                        normalize_ensemble = normalize_ensemble,
                        epochs=epochs, nepochs=2)

    elif(algorithm == "lof"):
        print("[DEBUG] Parameters: " + str(params))

        novelty = params["novelty"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = lof(novelty = novelty,
                    optimizer = optimizer,
                    loss = loss)

    elif(algorithm == "lscp"):
        print("[DEBUG] Chosen model:")
        print("[DEBUG] Parameters: " + str(params))

        #model = lof()

    elif(algorithm == "lunar"):
        print("[DEBUG] Chosen model:")
        print("[DEBUG] Parameters: " + str(params))

        #model = lof()

    elif(algorithm == "ocsvm"):
        print("[DEBUG] Parameters: " + str(params))

        kernel = params["kernel"]
        gamma = params["gamma"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = ocsvm(kernel = kernel,
                    gamma = gamma,
                    optimizer = optimizer,
                    loss = loss)

    elif(algorithm == "rod"):
        print("[DEBUG] Parameters: " + str(params))

        contamination = params["contamination"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = rod(contamination = contamination,
                    optimizer = optimizer,
                    loss = loss)

    elif(algorithm == "suod"):
        print("[DEBUG] Parameters: " + str(params))

        contamination = params["contamination"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = suod(contamination = contamination,
                    optimizer = optimizer,
                    loss = loss)

    elif(algorithm == "vae"):
        print("[DEBUG] Chosen model: Variational Auto-Encoder (VAE)")
        print("[DEBUG] Parameters: " + str(params))

        contamination = params["contamination"]
        gamma = params["gamma"]
        capacity = params["capacity"]
        epochs = params["epochs"]

        optimizer = params["optimizer"]
        loss = params["loss"]

        model = vae(contamination = contamination,
                        gamma = gamma,
                        capacity = capacity,
                        epochs = epochs,
                        optimizer = optimizer,
                        loss = loss)

    else:
        print("[DEBUG] Chosen model: Decision Tree")
        print("[DEBUG] Parameters: " + str(params))

        model = decisiontree()

    return model

#Utility function for the extraction of predictions from file with the format "'pred[0]' \n 'pred[1] \n... \n pred[N]"
def extract_list(file_path):
    
    # Initialize the list of lists
    obj_list = []

    # Read the file line by line and extract lists
    with open(file_path, 'r') as file:
        for line in file:
            element = line.strip()
            if('pred' in file_path): element = float(element)
            obj_list.append(element)

    # Print the extracted list of lists
    return obj_list

def read_file(file, format):      #Utility function to avoid useless nested if-then-else
    
    if(format == "csv"):
        df = pandas.read_csv(file, sep=",")

    elif(format == "parquet"):
        df = pandas.read_parquet(file)

        #Currently every parquet file is treated as it is a temporal packet series
        statistics_are_extracted = False

        if(not statistics_are_extracted):

            print("[DEBUG] Extracting statistics for CSV generation...")
            features = ['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL']                 #temporarily for Kitsune, IoT-23, TON-IoT
            df = get_statistics(df, features=features, num_packets=10)
            filename = file.split('/')[1]                   #data/nome_dataset/file.csv

            #Generating CSV in order not to repeat 'get_statistics' function in the future
            df.to_csv('data/' + filename + '/' + filename + '.csv', sep=',', index=False)

    return df, df.columns[:-1]


def search_models(client):
    page_size = 1000  # Number of results per page
    page_token = None  # Token for the next page, initially set to None

    # List to store all registered models
    all_models = []

    # Iterate over pages of results
    while True:
    # Search for registered models
        models = client.search_registered_models(max_results=page_size, page_token=page_token)
        print(len(models))

        # Add models from this page to the list
        all_models.extend(models)

        # Check if there are more pages
        if len(models) < page_size:
            break

        # Get the token for the next page
        page_token = models[-1].name

    model_names = [model.name for model in all_models]
    return model_names


def binarize(y, ds):
    f = open("conf.json")
    conf = json.load(f)
    y = [0 if label == conf["available_datasets"][ds]["benign_label"] else 1 for label in y]
    #y[y > 0] = 1        #0 is the BENIGN class, we choose 1 for encoding the MALWARE class label
    return y


def filter_classes(X, y, classes):

    indices = []

    for i, label in enumerate(y):
        if(label in classes): indices.append(i)

    X_filtered = list(X[indices])
    y_filtered = list(y[indices])

    return X_filtered, y_filtered


def remove_duplicate(Xtest, ytest):

    Xtest_nodup, indices = np.unique(Xtest, axis=0, return_index=True)      #careful to duplicate with different labels
    ytest_nodup = ytest[indices]

    return Xtest_nodup, ytest_nodup

def validate_thresholds(thresholds):

    perc_metrics = ["AUC", "pAUC", "avg_f1-score", "avg_precision", "avg_recall", "accuracy"]
    #Fix: currently, metrics for MD and AD specific cases are not distinguishable

    valid = True
    unknown_metrics = []

    for t in thresholds:

        if(t in perc_metrics):
            if(thresholds[t] < 0 or thresholds[t] > 1):
                print("[DEBUG] Invalid value of " + t + " (" + str(thresholds[t]) + ")")
                valid = False

        else:
            print("[DEBUG] Ignored unknown metric " + t)
            unknown_metrics.append(t)

    for m in unknown_metrics: del thresholds[m]

    return thresholds, valid