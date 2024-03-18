import os.path

import numpy as np
import pandas as pd
import scipy
from scipy.sparse.linalg import expm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import resample
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
import math
from tqdm import tqdm

from .datasets_info import *
from .extract_statistics import get_statistics


def load_data(dataset_name='iot-23', num_packets=20, seed=0, num_folds=0, target_fold=0):
    if dataset_name == 'kdd':
        assert num_folds == 0, 'NSL-KDD does not support folding, but the default hold-out.'

        train_fn = 'data/NSL-KDD/KDDTrain+.txt'
        test_fn = 'data/NSL-KDD/KDDTest+.txt'
        df_train = pd.read_csv(train_fn, header=None)
        df_test = pd.read_csv(test_fn, header=None)
        X_train = df_train.iloc[:, :-2].values
        y_train = df_train.iloc[:, -2].values
        X_test = df_test.iloc[:, :-2].values
        y_test = df_test.iloc[:, -2].values
    else:
        if dataset_name == 'iot-23':
            base_path = 'data/IoT-23/'
            path = base_path + 'iot23-stats_%d-pkts.parquet' % num_packets
        else:  # dataset_name == 'kitsune':
            base_path = 'data/Kitsune/'
            path = base_path + 'kitsune-stats_%d-pkts.parquet' % num_packets

        if not os.path.exists(path):
            print('Preprocessing raw parquet to generate statistics...')
            from glob import glob
            raw_parquet = pd.read_parquet(glob(os.path.join(base_path, 'dataset*.parquet'))[0])
            stats_parquet = get_statistics(
                raw_parquet, features=['PL', 'IAT', 'TTL', 'WIN'], label_col='LABEL', num_packets=num_packets)
            stats_parquet.to_parquet(path)

        df = pd.read_parquet(path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        if num_folds:
            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            train_index, test_index = list(skf.split(X, y))[target_fold]
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

            stored_train_fn = path.replace(
                '.parquet', '_seed_%d_train_%d-%dfold.parquet' % (seed, target_fold, num_folds))
            stored_test_fn = path.replace(
                '.parquet', '_seed_%d_test_%d-%dfold.parquet' % (seed, target_fold, num_folds))
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, stratify=y)

            stored_train_fn = path.replace('.parquet', '_seed_%d_train.parquet' % seed)
            stored_test_fn = path.replace('.parquet', '_seed_%d_test.parquet' % seed)

        # Storing training set if not present, else checking the current training set matches the stored.
        if not os.path.exists(stored_train_fn):
            pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1),
                         columns=df.columns).to_parquet(stored_train_fn)
        else:
            stored_train = pd.read_parquet(stored_train_fn)
            assert (stored_train.values == np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)).all()
            print('Training set check passed.')

        # Storing test set if not present, else checking the current test set matches the stored.
        if not os.path.exists(stored_test_fn):
            pd.DataFrame(np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1),
                         columns=df.columns).to_parquet(stored_test_fn)
        else:
            stored_test = pd.read_parquet(stored_test_fn)
            assert (stored_test.values == np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)).all()
            print('Test set check passed.')

    return X_train, y_train, X_test, y_test


def preprocess_data(X_train, y_train, X_test, y_test, dataset_name='iot-23', drop_duplicates=False):
    if dataset_name == 'kdd':
        cat_indexes = [i for i, v in enumerate(X_train[0]) if isinstance(v, str)]
        for cat_index in cat_indexes:
            oe = OrdinalEncoder()
            X_train[:, cat_index] = oe.fit_transform(X_train[:, cat_index].reshape(-1, 1)).squeeze()
            X_test[:, cat_index] = oe.transform(X_test[:, cat_index].reshape(-1, 1)).squeeze()

    mms = MinMaxScaler()
    X_train = mms.fit_transform(X_train)
    X_test = mms.transform(X_test)

    y_train = np.vectorize(datasets_label_encoder[dataset_name].get)(y_train)
    y_test = np.vectorize(datasets_label_encoder[dataset_name].get)(y_test)

    if drop_duplicates:
        no_duplicated = ~pd.concat((X_train, y_train), axis=1).duplicated()
        X_train, y_train = X_train[no_duplicated], y_train[no_duplicated]

        no_duplicated = ~pd.concat((X_train, y_train), axis=1).duplicated()
        X_test, y_test = X_test[no_duplicated], y_test[no_duplicated]

    return X_train, y_train, X_test, y_test


def get_training_index(y_train, detection_mode='anomaly', poisoning_ratio=0., poisoning_mode='stratified',
                       poisoning_batch_size=0.001):
    if detection_mode != 'anomaly':
        assert poisoning_ratio == 0, 'Poisoning is supported only for "anomaly" detection mode'

    if detection_mode == 'anomaly':
        # Retrieve only benign samples
        training_index = np.where(y_train == 0)[0]
    elif detection_mode == 'attack':
        # Retrieve only attack samples
        training_index = np.where(y_train != 0)[0]
    else:  # detection_mode == 'misuse'
        # Retrive both benign and attack samples
        training_index = list(range(len(y_train)))

    if poisoning_ratio:
        print('Poisoning the data with %.2f%% training set size...' % (poisoning_ratio * 100))
        # The poisoning candidate indexes are selected
        if poisoning_mode == 'stratified':
            poisoning_training_cadidates = np.where(y_train != 0)[0]
        else:
            poisoning_training_cadidates = np.where(y_train == int(poisoning_mode))[0]
        # Set the number of poisoning samples to select
        n_poisoning_samples = min(math.ceil(len(training_index) * poisoning_ratio), len(poisoning_training_cadidates))
        # Compute pet-batch poisoning samples number
        num_batches = math.ceil(poisoning_ratio / poisoning_batch_size)
        n_poisoning_samples_per_batch = [n_poisoning_samples // num_batches for _ in range(num_batches)]
        for i in range(n_poisoning_samples % num_batches):
            n_poisoning_samples_per_batch[i] += 1
        # Poisoning sample selection is conducted per-batch, in order to preserve, fixed the seed,
        # the order of selected samples
        for n_samples in n_poisoning_samples_per_batch:
            # To track selected candidates
            not_chosen_index = range(len(poisoning_training_cadidates))
            # Selection of n_samples storing the original index and the candidate index
            poisoning_training_index, chosen_index = resample(
                poisoning_training_cadidates, not_chosen_index, n_samples=n_samples, replace=False,
                stratify=y_train[poisoning_training_cadidates] if poisoning_mode == 'stratified' else None)
            # Removal via chosen_index of candidates and chosen sample indexes
            poisoning_training_cadidates = np.delete(poisoning_training_cadidates, chosen_index)
            # Extending the current training index with selected samples
            training_index = np.concatenate((training_index, poisoning_training_index))
    return training_index


def n_features_selection(x, y=None, method='amgm', feature_mode='max'):
#     from utils import get_xp
#     # Loading NumPy or CuPy if available
#     np = get_xp()

    num_feats = [x.shape[1]] + list(range(5, x.shape[1], 5))[::-1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_norm = scaler.fit_transform(x)

    if method == 'amgm':
        def amgm(x):
            """
            paper: Efficient feature selection filters for high-dimensional data
            """
            return 1 / len(x) * 1 / np.exp(np.mean(x)) * np.sum(np.exp(x))

        scores = [amgm(feat) for feat in X_norm.T]
        sorted_indexes = list(np.argsort(scores))
        if feature_mode == 'max':
            sorted_indexes = sorted_indexes[::-1]

    elif method == 'lse':
        def construct_W(X, neighbour_size=5, t=1):
            S = kneighbors_graph(X, neighbour_size + 1, mode='distance',
                                 metric='euclidean')  # sqecludian distance works only with mode=connectivity  results were absurd
            S = (-1 * (S * S)) / (2 * t * t)
            S = S.tocsc()
            S = expm(S)  # exponential
            S = S.tocsr()
            # [1]  M. Belkin and P. Niyogi, “Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering,” Advances in Neural Information Processing Systems,
            # Vol. 14, 2001. Following the paper to make the weights matrix symmetrix we use this method
            bigger = np.transpose(S) > S
            S = S - S.multiply(bigger) + np.transpose(S).multiply(bigger)
            return S

        def LaplacianScore(X, neighbour_size=5, t=1):
            W = construct_W(X, t=t, neighbour_size=neighbour_size)
            n_samples, n_features = np.shape(X)

            # construct the diagonal matrix
            D = np.array(W.sum(axis=1))
            D = scipy.sparse.diags(np.transpose(D), [0])
            # construct graph Laplacian L
            L = D - W.toarray()

            # construct 1= [1,···,1]'
            I = np.ones((n_samples, n_features))

            # construct fr' => fr= [fr1,...,frn]'
            Xt = np.transpose(X)

            # construct fr^=fr-(frt D I/It D I)I
            t = np.matmul(np.matmul(Xt, D.toarray()), I) / np.matmul(np.matmul(np.transpose(I), D.toarray()), I)
            t = t[:, 0]
            t = np.tile(t, (n_samples, 1))
            fr = X - t

            # Compute Laplacian Score
            fr_t = np.transpose(fr)
            Lr = np.matmul(np.matmul(fr_t, L), fr) / np.matmul(np.dot(fr_t, D.toarray()), fr)

            return np.diag(Lr)

        def distanceEntropy(d, mu=0.5, beta=10):
            """
            As per: An Unsupervised Feature Selection Algorithm: Laplacian Score Combined with
            Distance-based Entropy Measure, Rongye Liu
            """
            if d <= mu:
                result = (np.exp(beta * d) - np.exp(0)) / (np.exp(beta * mu) - np.exp(0))
            else:
                result = (np.exp(beta * (1 - d)) - np.exp(0)) / (np.exp(beta * (1 - mu)) - np.exp(0))
            return result

        def lse(data, ls=None):
            """
            This method takes as input a dataset, its laplacian scores for all features
            and applies distance based entropy feature selection in order to identify
            the best subset of features in the laplacian sense.
            """
            if ls is None:
                ls = LaplacianScore(data)

            orderedFeatures = np.argsort(ls)
            scores = {}
            for i in range(2, len(ls)):
                selectedFeatures = orderedFeatures[:i]
                selectedFeaturesDataset = data[:, selectedFeatures]
                d = pairwise_distances(selectedFeaturesDataset, metric='euclidean')
                beta = 10
                mu = 0.5

                d = MinMaxScaler().fit_transform(d)
                e = np.vectorize(distanceEntropy)(d, mu, beta)
                e = MinMaxScaler().fit_transform(e)
                totalEntropy = np.sum(e)
                scores[i] = totalEntropy

            bestFeatures = orderedFeatures[:list(scores.keys())[np.argmin(scores.values())]]
            return bestFeatures

    else:
        raise NotImplementedError('Method %s is not supported.' % method)

    if method in ['amgm']:  # Raking-based methods
        selected_features = []
        for n_feats in num_feats:
            selected_features.append(sorted_indexes[:n_feats])
    else:
        pass

    features = dict()
    for i in range(len(selected_features)):
        features[num_feats[i]] = {
            'selected_features': int
        }
        features[num_feats[i]] = selected_features[i]

    return features
