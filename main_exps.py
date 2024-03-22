#BATCH OF EXPERIMENTS

import sys
import os
import json
import itertools
import mlflow
from run_pipeline import run_pipeline
import hashlib
import pandas as pd
from loggers.plot import plot_avg_roc

#MAIN
def main(argv):
    
    #exps_setup.json file taken as input for the script
    f = open('exp_setup.json')
    exp_setup = json.load(f)     #dict containing our exp. session configuration

    f = open('conf.json')
    conf = json.load(f)

    f = open('models.json')
    models_info = json.load(f)

    #extract data for our configuration
    exp_name = exp_setup["general-parameters"]["exp-name"]
    dataset_list = exp_setup["ingest"]["dataset-list"]
    algorithms = exp_setup["train"]["algorithms"]
    random_state_list = eval(exp_setup["general-parameters"]["random-state"])
    tracking_uri = exp_setup["general-parameters"]["tracking-uri"]

    for ds in dataset_list:
        if not ds in conf["available_datasets"]:
            print("[ERROR] A data set was not found in the list of the available ones")
            return

    if(not 'exp_reports' in os.listdir()): os.mkdir('exp_reports')
    if(not exp_name in os.listdir('exp_reports')): os.mkdir('exp_reports/' + exp_name)

    #From "task" field of JSON config file we infer:
    # - what model can be used
    # - additional options, like "binarize-labels"
    task = exp_setup["preprocess"]["task"]

    options = {
        "k_fold" : exp_setup["split"]["k-fold"],
        "samples_barrier" : exp_setup["preprocess"]["samples-barrier"],
        "no_duplicate" : 0,
        #"no_duplicate" : exp_setup["options"]["no-duplicate"],
        "num_folds" : exp_setup["split"]["num-folds"],
        "explain" : exp_setup["evaluate"]["explain"],
        "split_proportion" : exp_setup["split"]["split-portion"]
    }

    runs = 0

    #algorithms is a list of dictionaries, every one with a ML algorithm as the key
    #
    #The idea is to realize a list of configs to pass to the main of ML pipeline

    configs = {}

    for alg in algorithms.keys():

        configs[alg] = []

        hyperparams = list(algorithms[alg].keys())
        values_list = list(algorithms[alg].values())
        configs_list = list(itertools.product(*values_list))

        for config in configs_list:

            c_dict = {}

            for j, param in enumerate(hyperparams):
                c_dict[param] = config[j]

            configs[alg].append(c_dict)

    
    if(task == "bMD"):
        options["binarize_labels"] = 1
    elif(task == "mMD"):
        options["binarize_labels"] = 0
    elif(task == "AD"):
        options["binarize_labels"] = 1
    else:
        print("[ERROR] Task not recognized.")

    for dataset in dataset_list:
        for alg in algorithms.keys():
            for config in configs[alg]:

                default_config = True      #check on the default configuration of the algorithm

                for p in config:
                    if(config[p] != models_info["algorithm"][alg]["hyper-parameters"][p]): default_config = False

                #Filling config with params not used 
                for p in models_info["algorithm"][alg]["hyper-parameters"]:
                    if not p in config:
                        config[p] = models_info["algorithm"][alg]["hyper-parameters"][p]

                #TO FIX: Re-ordering hyper-params in order to pass them to hash function in the always in the same order

                for seed in random_state_list:

                    runs = runs + 1
                    run_name = dataset + "_" + alg + "_" + "seed_" + str(seed) + "_task_" + task

                    print("[DEBUG] Run number " + str(runs) + " of the current exp. batch")

                    #summary of charateristics of the run
                    print("[DEBUG] Summary")
                    print("[DEBUG] Dataset: " + dataset)
                    print("[DEBUG] Algorithm: " + alg)
                    print("[DEBUG] Run name: " + run_name)
                    print("[DEBUG] Options: " + str(options))
                    print("[DEBUG] Configuration: " + str(config) + (" (default)" if default_config else ""))
                    print("[DEBUG] Seed: " + str(seed))
                    print("[DEBUG] Task: " + task)

                    #Note that a run has its own ID number, maybe not needed the insertion of a hash code to differentiate run names...
                    hash_input = str(config)
                    run_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
                    run_name = run_name + "_config_" + run_hash

                    run_pipeline(
                        dataset = dataset,
                        algorithm = alg,
                        exp_name = exp_name,
                        run_name = run_name,
                        label = conf["available_datasets"][dataset]["labels"][0],            #to fix for multi-level label setup               
                        options = options,
                        params = config,
                        thresholds = exp_setup["evaluate"]["thresholds"],
                        tracking_uri = tracking_uri,
                        random_state = seed,
                        task = task
                    )

                    print("-"*80)
                    print("-"*80)

    print("[DEBUG] Batch completed. Reporting results...")

    #REPORT OF THE EXP. BATCH COMPILING
    exp_id = mlflow.get_experiment_by_name(exp_name).experiment_id
    runs = mlflow.search_runs(experiment_ids=exp_id)
    filename = 'exp_reports/' + exp_name + '/' + exp_name + ".csv"
    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(filename, index=False)
    print("[DEBUG] Executing command: python3 get_artifacts_size.py " + filename)       #
    os.system("python3 get_artifacts_size.py " + filename)
    os.remove(filename)
    

    #CONSTRUCTION OF PLOT OF DIFFERENT TYPOLOGY
    #The idea is to configure them as a comparation at different levels
    # (between models, between datasets, between different configurations of the same models etc.)

    if(task == "AD"):

        print("[DEBUG] Plotting average ROCs...")

        for ds in dataset_list:
            plot_avg_roc(exp_id, exp_name, level="algs", ds=ds)
        for a in algorithms:
            plot_avg_roc(exp_id, exp_name, level="data", alg=a)


    return


if __name__ == '__main__':
    main(sys.argv)