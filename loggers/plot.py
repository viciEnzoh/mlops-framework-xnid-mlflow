import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn as sns
import numpy as np
import mlflow
from utils import extract_list
import itertools


def plot_roc(y_true, y_score):
    fpr, tpr, thrs = roc_curve(y_true, y_score, pos_label=1)
    indices=np.where(fpr>=0.01)
    index=np.min(indices)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, c='firebrick', zorder=4)
    
    AUC = auc(fpr, tpr)

    if(len(np.unique(y_true)) > 2): pAUC = None
    else: pAUC = roc_auc_score(y_true, y_score, max_fpr = .01)

    tpr_1 = tpr[index]
    stampa = 'Area = ' + str(round(AUC, 4))
    stampa2 = 'TPR_1 = ' + str(round(tpr_1,4))
    if(len(np.unique(y_true)) <= 2): stampa3 = 'pAUC = ' + str(round(pAUC,4))
    ax.fill_between(fpr, tpr, color='salmon')
    plt.text(0.4, 0.2, stampa, fontsize=12)
    plt.text(0.4, 0.13, stampa2, fontsize=12)
    if(len(np.unique(y_true)) <= 2): plt.text(0.4, 0.06, stampa3, fontsize=12)
    plt.axvline(x=0.01, color='grey', linestyle='--')
    ax.plot([0, 1], [0, 1], c='k', ls='--', zorder=3)
    ax.set_xlim(.001, 1)
    ax.set_ylim(0, 1)
    ax.set_xscale(value="log")
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    plt.tight_layout()

    plt.savefig('roc.pdf')
    plt.savefig('roc.png')
    plt.close(fig)

    return AUC, pAUC


def plot_cm(y_true, y_pred, labels, normalize='true', train_histo=None):
    """
    normalize == 'true', normalization on rows: putting recall on the main diagonal, cm/sum(cm, axis=1).reshape(-1,1)
    normalize == 'pred', normalization on cols: putting precision on the main diagonal, cm/sum(cm, axis=0)
    normalize == 'all, cm/sum(cm)
    """

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    num_classes = len(set(y_true))

    if normalize == 'true':
        from sklearn.metrics import recall_score
        _cm = confusion_matrix(y_true, y_pred, labels=labels)
        assert (cm == _cm / np.sum(_cm, axis=1).reshape(-1, 1)).all()
        assert (np.diag(cm) == recall_score(y_true, y_pred, average=None)).all()
    if normalize == 'pred':
        from sklearn.metrics import precision_score
        _cm = confusion_matrix(y_true, y_pred, labels=labels)
        assert (cm == _cm / np.sum(_cm, axis=0)).all()
        assert (np.diag(cm) == precision_score(y_true, y_pred, average=None)).all()
    if normalize == 'all':
        _cm = confusion_matrix(y_true, y_pred, labels=labels)
        assert (cm == _cm / np.sum(_cm)).all()

    vmax = None
    if normalize:
        vmax = 1.

    if train_histo is not None:
        fig, axes = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={'height_ratios': [1, 5]})
        x = list(range(num_classes))
        y_v, y_c = np.unique(train_histo, return_counts=True)
        y = [y_c[i] for i in np.argsort(y_v)]
        y = [0] * (num_classes - len(y)) + y
        sns.barplot(x=x, y=y, ax=axes[0], color='royalblue')
        axes[0].axis('off')
        ax = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    sns.heatmap(cm, vmin=.0, vmax=vmax, cmap='Reds', square=True, cbar=False, ax=ax, zorder=3)
    ax.set_xticks([v + .5 for v in range(num_classes)])
    ax.set_xticklabels([c for c in labels], rotation=90)
    ax.set_yticks([v + .5 for v in range(num_classes)])
    ax.set_yticklabels([c for c in labels], rotation=0)
    ax.xaxis.set_tick_params(width=5, color='k', labelsize=14)
    ax.yaxis.set_tick_params(width=5, color='k', labelsize=14)

    ax.set_xlabel('Preds', fontsize=16)
    ax.set_ylabel('Trues', fontsize=16)

    plt.tight_layout()
    
    plt.savefig('confusion_matrix.pdf')
    plt.savefig('confusion_matrix.png')
    plt.close(fig)


def plot_avg_roc(exp_id, exp_name, level="algs", ds="NSL-KDD", alg="iforest"):

    runs_list = mlflow.search_runs(exp_id)
    plot_dict = {}
    fig, ax = plt.subplots()

    #setting a dictionaries containing lists
    #level = "algs": comparation for a data set of AD algorithms
    if(level == "algs"):
        for a in np.unique(runs_list.loc[runs_list["tags.dataset"] == ds, "tags.algorithm"]):          #chose dataset

            plot_dict[a] = {}
            plot_dict[a]["ytrue"] = []
            plot_dict[a]["yscore"] = []

            #select runs for that specific algorithm
            for run_id in runs_list.loc[(runs_list["tags.algorithm"] == a) & (runs_list["tags.dataset"] == ds), "run_id"]:

                artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id)
                ytrue = extract_list(artifact_path + "/ytest.txt")
                yscore = extract_list(artifact_path + "/ypred.txt")

                plot_dict[a]["ytrue"].append(ytrue)
                plot_dict[a]["yscore"].append(yscore)

    #level = "data": comparation for an AD algorithm among datasets
    if(level == "data"):
        for d in np.unique(runs_list.loc[runs_list["tags.algorithm"] == alg, "tags.dataset"]):            #chose algorithm

            plot_dict[d] = {}
            plot_dict[d]["ytrue"] = []
            plot_dict[d]["yscore"] = []

            #select runs for that specific algorithm
            for run_id in runs_list.loc[(runs_list["tags.algorithm"] == alg) & (runs_list["tags.dataset"] == d), "run_id"]:        #where tags.algorithm = a // runs_list.loc[runs_list[["tags.algorithm"] == a, "run_id"]

                artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id)
                ytrue = extract_list(artifact_path + "/ytest.txt")
                yscore = extract_list(artifact_path + "/ypred.txt")

                plot_dict[d]["ytrue"].append(ytrue)
                plot_dict[d]["yscore"].append(yscore)

    #level = "conf": comparation for an AD algorithm on a single dataset for different configs
    if(level == "conf"): pass

    #Obtaining lists of AD algorithms execution outputs
    base_colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    colors = itertools.cycle(base_colors)
    base_linestyles = ['-', '--', '-.', ':']
    linestyles = itertools.cycle(base_linestyles)

    for element in plot_dict:       #element in {'dataset', 'algorithm'}

        y_true_list = plot_dict[element]["ytrue"]
        y_score_list = plot_dict[element]["yscore"]

        # Compute ROC curves and AUCs for each seed
        fprs = []
        tprs = []
        aucs = []

        linewidth = 3
        color = next(colors)
        linestyle = next(linestyles)

        for y_true, y_score in zip(y_true_list, y_score_list):
            fpr, tpr, _ = roc_curve(y_true, y_score, pos_label='MALWARE')
            roc_auc = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            aucs.append(roc_auc)

        # Interpolate ROC curves to have same number of points
        mean_fpr = np.linspace(0, 1, 100)
        tpr_interp = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fprs, tprs)]
        mean_tpr = np.mean(tpr_interp, axis=0)
        mean_auc = np.mean(aucs)
        
        std_tpr = np.std(tpr_interp, axis=0)

        # Plot average ROC curve
        plt.plot(mean_fpr, mean_tpr, label=element, linestyle=linestyle, color=color, linewidth=linewidth, zorder=4)
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.25)

    #plt.axvline(x=0.01, color='grey', linestyle='--')
    plt.xlim(.001, 1)
    plt.ylim(0, 1)
    plt.xlabel('FPR', fontsize=24)
    plt.ylabel('TPR', fontsize=24)
    #plt.xscale(value="log")
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)
    #plt.title('')

    plt.plot([0, 1], [0, 1], color='k', linestyle='--', label='chance', zorder=1)
    l = plt.legend(fontsize=22, loc=(1.1, 0.25),ncol=1, frameon=False)
    for line in l.get_lines():
        line.set_linewidth(linewidth)

    plt.tight_layout()
    
    plt.savefig('exp_reports/' + exp_name + '/' + (ds if level == "algs" else alg) + '_avg_roc.pdf')
    plt.savefig('exp_reports/' + exp_name + '/' + (ds if level == "algs" else alg) + '_avg_roc.png')
    plt.close(fig)
