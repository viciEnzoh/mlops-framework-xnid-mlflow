from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve, roc_auc_score
from mlflow.models.signature import infer_signature
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pickle
import json

from loggers.plot import plot_roc, plot_cm

import numpy as np
import shap

from utils import remove_duplicate



def evaluate(model, Xtest, ytest, algorithm=None, no_dup=0, binarize_labels=1, explain=0, features=[], Xtrain=None, ytrain=None, le=None, random_state=0, kfold_mode=False):       #ytrain if histograms on CM are wished

    performance = {}
    report = {}
    f = open('models.json')
    models_info = json.load(f)

    print("[DEBUG] Test set dimension: " + str(len(Xtest)))

    if(no_dup):
        Xtest, ytest = remove_duplicate(Xtest, ytest)
        print("[DEBUG] Test set dimension after duplicate removing: " + str(len(Xtest)))


    if(models_info["algorithm"][algorithm]["learning"] == "supervised"):

        #PREDICTIONS COMPUTING
        ypred_prob = model.predict_proba(Xtest)
        ypred = [np.argmax(prediction) for prediction in ypred_prob]

        #CLASS DECODING
        #multiclass case
        if(not binarize_labels):
            #ypred
            class_encoding = [(i, np.unique(ytest)[i]) for i in range(len(np.unique(ytest)))]
            ypred = [class_encoding[prediction][1] for prediction in ypred]
            ypred = le.inverse_transform(ypred)
            #ytest
            ytest = le.inverse_transform(ytest)

        else:
            ypred = ['BENIGN' if label == 0 else 'MALWARE' for label in ypred]
            ytest = ['BENIGN' if label == 0 else 'MALWARE' for label in ytest]


        report = classification_report(ytest, ypred, output_dict=True)
        print()
        print(classification_report(ytest, ypred))

        if(explain):            
            print("[DEBUG] Calculating the SHAP values...")
            index = np.arange(len(Xtrain))
            index = np.random.choice(index, size=250, replace=False)
            Xshap = np.array(Xtrain)[index]
            explainer = shap.Explainer(model.model, masker=Xshap, njobs=20, approximate=True) #shap.sample(np.array(Xtrain), nsamples=500, random_state=random_state)
            # explainer = shap.Explainer(model.model, masker=shap.sample(np.array(Xtrain), nsamples=500, random_state=random_state), njobs=20, approximate=True)
            shap_values = explainer.shap_values(np.array(Xshap))
            shap.summary_plot(shap_values, np.array(Xtest), feature_names=features, max_display=10, plot_size=(16,9), show=False)

            handle, labels = plt.gca().get_legend_handles_labels()
            converted_labels = le.inverse_transform(model.model.classes_)
            plt.xlabel("")
            plt.legend(handle, converted_labels, frameon=False, fontsize=18, loc=(.7, .15))#, ncols=5)
            plt.savefig("shap_summary_plot.png")
            plt.savefig("shap_summary_plot.pdf")
            with open('shap_values.pkl', 'wb') as f:
                pickle.dump(shap_values, f)


        print("[DEBUG] Classification report of the model...")

        performance['accuracy'] = report['accuracy']
        performance['macro_avg_precision'] = report['macro avg']['precision']
        performance['macro_avg_recall'] = report['macro avg']['recall']
        performance['macro_avg_f1-score'] = report['macro avg']['f1-score']
        performance['weighted_avg_precision'] = report['weighted avg']['precision']
        performance['weighted_avg_recall'] = report['weighted avg']['recall']
        performance['weighted_avg_f1-score'] = report['weighted avg']['f1-score']

        labels = np.unique(ytest)
        if(not kfold_mode): plot_cm(ytest, ypred, labels, normalize='true', train_histo=None)

    elif(models_info["algorithm"][algorithm]["learning"] == "unsupervised"):

        ypred = model.score(Xtest)

        print("[DEBUG] ROC curve generation...")
        if(not kfold_mode):
            AUC, pAUC = plot_roc(ytest, ypred)
        else:
            fpr, tpr, thrs = roc_curve(ytest, ypred, pos_label=1)
            AUC = auc(fpr, tpr)
            pAUC = roc_auc_score(ytest, ypred, max_fpr = .01)

        performance['AUC'] = AUC
        performance['pAUC'] = pAUC

    return performance, report, ypred