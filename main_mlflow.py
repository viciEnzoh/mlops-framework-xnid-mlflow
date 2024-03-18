#PIPELINE MLFLOW

import mlflow
from mlflow import MlflowClient
import argparse
import sys
import os
import hashlib
import json

from run_pipeline import run_pipeline

from classifiers.classifiers_params import classifiers_params_info as cpi, eval_list as el, sup_learning as sl, unsup_learning as ul

from pipeline.pipeline import Pipeline

from loggers.log import log_artifacts, log_metrics, log_model, log_options

#MAIN
def main(argv):

    #GLOBAL VARIABLES SETTING

    #SHELL PARAMETERS PARSING
    print("[DEBUG] Parsing arguments passed by the shell...")
    parser = argparse.ArgumentParser(description='MLFlow Pipeline example')

    # > register (true if registration of the model is wished)
    parser.add_argument('--register-model', type=bool, default=False, required=False,
                        help='Indicates if the insertion of the model in registry is performed, indipendently from validation results (default=%(default)s)')
    
    # > tracking-uri (set of the tracking server URI)
    parser.add_argument('--tracking-uri', type=str, default="http://localhost:5000", required=False,
                        help='Indicates the URI of the tracking server (default=%(default)s)')

    # > experiment-name (set of the name of the experiment)
    parser.add_argument('--exp-name', type=str, default="Default", required=False,
                        help='Indicates the name of the chosen experiment (default=%(default)s)')
    
    # > run-name (set of the name of the single run)
    parser.add_argument('--run-name', type=str, default="prova", required=False,
                        help='Indicates a symbolic name for the run (default=%(default)s)')

    # > model-name (if I'm searching for a specific pre-registered model)

    # > split-proportion
    parser.add_argument('--split-proportion', type=str, default="[0.7, 0.3]", required=False,
                        help='Indicates the proportion of training/testing set sizes for esplicit splitting (default=%(default)s)')

    # > dataset-name
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='Indicates the name of the folder where reside the data set files')
    
    # > random-state
    parser.add_argument('--random-state', type=int, default=10, required=False,
                        help='Indicates the seed of the run that is going to be launched (default=%(default)s)')
    
    # > k-fold (0: no (default), 1: yes, 2: only k-fold validation without train+test on a model)
    parser.add_argument('--k-fold', type=int, default=0, required=False,
                        choices=[0, 1, 2],
                        help='Indicates experiment mode among train+test, k-fold or both (default=%(default)s = train+test)')

    # > num-folds
    parser.add_argument('--num-folds', type=int, default=10, required=False,
                        help='Indicates the number of folds in stratified K-fold validation (default=%(default)s)')
    # > target-fold
    parser.add_argument('--target-fold', type=int, default=0, required=False,
                        help='Indicates the index of the target fold in stratified K-fold validation (required if KxV wished) (default=%(default)s)') # ###
    
    # > no-duplicate
    parser.add_argument('--no-duplicate', type=int, default=0, required=False,
                        choices=[0, 1],
                        help='Sets if not to consider duplicate samples in testing set (default=%(default)s)')
    
    # > binarize-labels
    parser.add_argument('--binarize-labels', type=int, default=0, required=False,
                        choices=[0, 1],
                        help='Sets if to let labels be binary (0 -> Benign, 1 -> Malware) (default=%(default)s)')
    
    # > explain
    parser.add_argument('--explain', type=int, default=0, required=False,
                        choices=[0, 1],
                        help='Sets if to explain features and produce a summary for explainations to log (default=%(default)s)')

    # > samples-barrier
    parser.add_argument('--samples-barrier', type=int, default=300, required=False,
                        help='Sets the minimum amount of samples a class must be represented by (default=%(default)s)')

    # > algorithm
    parser.add_argument('--algorithm', type=str, default='decision-tree', required=False,
                        choices=sl+ul,
                        help='ML algorithm chosen for the experiment (default=%(default)s)')

    # > params
    parser.add_argument('--params', nargs='*', metavar="PARAMETER=VALUE", required=False, default=[],
                        help='Parameters for tuning chosen ML algorithm')

    # > thresholds
    parser.add_argument('--thresholds', nargs='*', metavar="METRIC=VALUE", required=False,
                        help='Thresholds for model validation')

    args, extra_args = parser.parse_known_args(argv)

    f = open('conf.json')
    conf = json.load(f)     #dict containing our exp. session configuration
    if(not args.dataset_name in conf["available_datasets"]):
        print("[ERROR] Dataset not present in list of available.")
        return

    #MLFLOW INFO EXTRACTION
    label = conf["available_datasets"][args.dataset_name]["labels"][0]        #Currently, we are not effectively addressing multi-label problems

    options = {
        "binarize_labels" : args.binarize_labels,
        "samples_barrier" : args.samples_barrier,
        "no_duplicate" : args.no_duplicate,
        "explain" : args.explain,
        "split_proportion" : eval(args.split_proportion),
        "k_fold" : args.k_fold,
        "num_folds" : args.num_folds
    }

    thresholds = args.thresholds
    random_state = args.random_state

    run_pipeline(
        dataset = args.dataset_name,
        algorithm = args.algorithm,
        exp_name = args.exp_name,
        run_name = args.run_name,
        label = label,                   
        options = options,
        params = args.params,
        thresholds = thresholds,
        tracking_uri = args.tracking_uri,
        random_state = random_state,
        task = None,                            ###
        single_run = True
    )
    
    return

if __name__ == '__main__':
    main(sys.argv)