
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

from datasets.datasets_info import *

import numpy as np

from utils import binarize, filter_classes

def preprocess(Xtrain, Xtest, ytrain, ytest, dataset_name, binarize_labels=0, samples_barrier=300):

    #For temporal series...
    #
    ##
    #
    if dataset_name == 'NSL-KDD':
        cat_indexes = [i for i, v in enumerate(Xtrain[0]) if isinstance(v, str)]

        print("[DEBUG] Ordinal Encoding...")

        for cat_index in cat_indexes:
            oe = OrdinalEncoder()
            Xtrain[:, cat_index] = oe.fit_transform(Xtrain[:, cat_index].reshape(-1, 1)).squeeze()
            Xtest[:, cat_index] = oe.transform(Xtest[:, cat_index].reshape(-1, 1)).squeeze()

    print("[DEBUG] MinMax Scaling...")
    mms = MinMaxScaler()

    Xtrain = mms.fit_transform(Xtrain)
    Xtest = mms.transform(Xtest)

    le = LabelEncoder()

    #Binarize if requested
    if(binarize_labels):
        print("[DEBUG] Binarizing labels as requested (positive class: not BENIGN)") 
        ytrain = binarize(ytrain, dataset_name)
        ytest = binarize(ytest, dataset_name)

    else:
        print("[DEBUG] Label Encoding...")
        le.fit(list(ytrain) + list(ytest))
        ytrain = le.transform(ytrain)
        ytest = le.transform(ytest)
        
        print("[DEBUG] Mapping of labels:")
        enc_classes = le.transform(le.classes_)
        for i, c in enumerate(le.classes_):
            print("[DEBUG] Class " + c + " --> " + str(enc_classes[i]))
        print()

        #ytrain = np.vectorize(datasets_label_encoder[dataset_name].get)(ytrain)
        #ytest = np.vectorize(datasets_label_encoder[dataset_name].get)(ytest)

    if(len(np.unique(ytrain)) > 2):           #Only in multiclass scenario
        #Counting samples per class (barrier = 300)
        train_classes, train_count = np.unique(ytrain, return_counts=True)
        test_classes, test_count = np.unique(ytest, return_counts=True)

        print('[DEBUG] Training set labels count')
        for i, data_class in enumerate(train_classes):
            print('[DEBUG] Label ' + str(le.inverse_transform([data_class])[0]) + ': ' + str(train_count[i]))
        print()

        print('[DEBUG] Testing set labels count')
        for i, data_class in enumerate(test_classes):
            print('[DEBUG] Label ' + str(le.inverse_transform([data_class])[0]) + ': ' + str(test_count[i]))
        print()

        #Two conditions for class elimination:
        # (1) presence of class in testing set that the classifier has not trained to predict
        # (2) presence of class in training set that are not being tested on
        # (3) the class samples number is not overcoming samples barrier
        
        class_count = {}

        for i, data_class in enumerate(train_classes):
            class_count[data_class] = train_count[i]

        for i, data_class in enumerate(test_classes):
            if(data_class in class_count.keys()):           #implicit filtering of classes in testing set that the classfiers cannot recognize (1)
                class_count[data_class] = class_count[data_class] + test_count[i]

        for i, data_class in enumerate(train_classes):
            if(not data_class in test_classes): class_count.pop(data_class)     #Condition (2)

        valid_classes = []

        print('[DEBUG] Samples barrier: ' + str(samples_barrier))

        for data_class in class_count:
            if(class_count[data_class] >= samples_barrier):
                print("[DEBUG] " + str(le.inverse_transform([data_class])[0]) + " is a valid class (" + str(class_count[data_class]) + " samples)")
                valid_classes.append(data_class)        #Condition (3)
            else:
                print('[DEBUG] Removing class ' + str(le.inverse_transform([data_class])[0]) + ' for not passing the samples barrier')

        print('[DEBUG] Valid classes: ' + str(le.inverse_transform(valid_classes)))
        print('[DEBUG] Valid classes (encoded): ' + str(valid_classes))

        print('[DEBUG] Training set dimension before sample barrier filtering: ' + str(len(Xtrain)))
        Xtrain, ytrain = filter_classes(Xtrain, ytrain, valid_classes)
        print('[DEBUG] Training set dimension after sample barrier filtering: ' + str(len(Xtrain)))

        print('[DEBUG] Testing set dimension before sample barrier filtering: ' + str(len(Xtest)))
        Xtest, ytest = filter_classes(Xtest, ytest, valid_classes)
        print('[DEBUG] Testing set dimension after sample barrier filtering: ' + str(len(Xtest)))

    return Xtrain, Xtest, ytrain, ytest, le

