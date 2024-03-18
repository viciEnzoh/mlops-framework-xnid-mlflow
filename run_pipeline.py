#PIPELINE MLFLOW

import mlflow
from mlflow import MlflowClient
import hashlib
import json
from pipeline.pipeline import Pipeline
from steps import kfold
from utils import validate_thresholds
from loggers.log import log_artifacts, log_metrics, log_model

def run_pipeline(
        dataset = None,
        algorithm = None,
        exp_name = None,
        run_name = None,
        label = None,                   
        options = None,
        params = None,
        thresholds = None,
        tracking_uri = None,
        random_state = None,
        task = None,
        single_run = False                                      #if a I want to launch a single run via console
):
    
    #SETTING MLFLOW ENVIRONMENT INFO
    mlflow.set_tracking_uri(tracking_uri)                       #setting the URI of the tracking server
    mlflow.set_experiment(experiment_name=exp_name)             #setting the name of the experiment (creating if it does not exist)
    
    with mlflow.start_run() as run:
    
        mlflow.set_tag("mlflow.runName", run_name)                  #setting the name of the single run
        mlflow.log_param('algorithm', algorithm)                    #logging the ML algorithm as a param for experimental results table view
        mlflow.log_param('proportion', options["split_proportion"]) #logging info regarding the partitioning of the dataset
        run_id = mlflow.active_run().info.run_id                    #extracting the current run ID

        #GLOBAL VARIABLES SETTING

        #MLFLOW INFO EXTRACTION
        print("[DEBUG] Setting MLflow experiment variables...")
        model_tags = {}                                         #potential tags: type of algorithm (classifier, detector)

        #General information tags
        model_tags["dataset"] = dataset
        model_tags["task"] = task
        model_tags["seed"] = random_state
        model_tags["algorithm"] = algorithm
        model_tags["proportion"] = options["split_proportion"]

        #Initialize a string for hashing function input
        hash_input = ""

        #PARAMETERS DISPATCHING
        #Re-ordering to standardize the order of presentation of hyper-params composing the name of the model
        #... and generating a model hash value to compact model name
        
        #Here params is a data structure already in the right format, it has only to be aligned with model_tags
        #args.params.sort()
        params_dict = {}

        f = open('models.json')
        models_info = json.load(f)

        if(single_run):         #via console
            for p in params:
                [key, value] = p.split("=")
                if(key in models_info["algorithm"][algorithm]["hyper-parameters"]):
                    
                    #eval is applicated to numerical type parameters in order to log them as numbers
                    if(key in models_info["eval_list"]):
                        params_dict[key] = eval(value)
                        model_tags[key] = eval(value)
                    else:
                        params_dict[key] = value
                        model_tags[key] = value

                    hash_input = hash_input + str(key) + "_" + str(value) + "_"

        else:                   #via batch script
            for key in params: params_dict[key] = params[key]

            for p in params_dict:
                if(p in models_info["algorithm"][algorithm]["hyper-parameters"]):                    
                    hash_input = hash_input + str(p) + "_" + str(params_dict[p]) + "_"


        #assigning default value for parameters missing in algorithm configuration (useful in case of single run, main_exps.py performs this internally)
        for p in models_info["algorithm"][algorithm]["hyper-parameters"]:
            if not p in params_dict:
                params_dict[p] = models_info["algorithm"][algorithm]["hyper-parameters"][p]
                model_tags[p] = models_info["algorithm"][algorithm]["hyper-parameters"][p]
                hash_input = hash_input + str(p) + "_" + str(params_dict[p]) + "_"


        mlflow.log_params(params_dict)                          #logging the parameters tuning the model for the run

        #Adding some info to hash generation
        if(single_run): task = ""
        hash_input = hash_input + task + "_" + str(options["split_proportion"]) + "_"

        model_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]

        #THRESHOLDS UNPACKING
        #to decide if to register a model
        thresholds_dict = {}
        
        if(single_run):

            for t in thresholds:
                [key, value] = t.split("=")
                thresholds_dict[key] = eval(value)

        else:
            thresholds_dict = thresholds

        thresholds_dict, valid_thresholds = validate_thresholds(thresholds_dict)
        for t in thresholds_dict:
            #print("[DEBUG] For " + t + " - value of threshold: " + str(thresholds_dict[t]))
            mlflow.log_param("threshold_" + t, thresholds_dict[t])
            model_tags["threshold_" + t] = thresholds_dict[t]
        if(not valid_thresholds): print("[DEBUG] Some invalid values of thresholds have been detected, those are going to be ignored")

        #name of the model
        model_name = dataset + "_" + algorithm + "_random_state_" + str(random_state) + "_" + model_hash

        #num_folds: default value is 10, if not inserted in options dict
        num_folds = options["num_folds"] if "num_folds" in options.keys() else 10

        #PIPELINE DEFINITION
        pipeline = Pipeline(dataset=dataset,
                            label=label,
                            steps=["ingest", "split", "preprocess", "transform", "train", "evaluate"],
                            algorithm=algorithm,
                            random_state = random_state,
                            params=params_dict,
                            no_dup = options["no_duplicate"],
                            binarize_labels = options["binarize_labels"],
                            samples_barrier = options["samples_barrier"],
                            explain=options["explain"],
                            split_proportion = options["split_proportion"],
                            num_folds = num_folds)
                            
        pipeline.steps()

        #INGEST
        #When the dataset chosen by the user is loaded in the pipe
        print("[DEBUG] Ingesting the chosen data set...")
        pipeline.ingest()
        #return          #To activate only get_statistics() function

        #SPLIT
        #When the dataset partitions (training, testing set) are made
        print("[DEBUG] Splitting the data set...")
        len_Xtrain, len_Xtest = pipeline.split()
        mlflow.log_param('train_set_length', len_Xtrain)
        mlflow.log_param('test_set_length', len_Xtest)
        model_tags["train_set_length"] = len_Xtrain
        model_tags["test_set_length"] = len_Xtest

        #PREPROCESS
        #When the data format is made more suitable to the pipeline
        print("[DEBUG] Preprocessing the data...")
        le = pipeline.preprocess()
        #return

        #K-FOLD option
        if(options["k_fold"]): pipeline.kfold()

        #Without performing k-fold X validation best_config is assumed to be the same as the unique config indicated by user :)

        #TRANSFORM
        #When the data acrosses state space transformation for model performance enhancement
        #print("[DEBUG] Transforming the data...")
        #pipeline.transform()

        #TRAIN [if required]
        #When the model is trained by getting in input the training part of the data set
        client = MlflowClient()
        already_registered_model = False

        filter_string = "tags.dataset = '" + dataset + "'"
        #filter_string = filter_string + " and tags.task = '" + task + "'"
        filter_string = filter_string + " and tags.algorithm = '" + algorithm + "'"
        filter_string = filter_string + " and tags.seed = '" + str(random_state) + "'"
        filter_string = filter_string + " and tags.samples_barrier = '" + str(options["samples_barrier"]) + "'"
        filter_string = filter_string + " and tags.proportion = '" + str(options["split_proportion"]) + "'"
        for p in params_dict:
            filter_string = filter_string + " and tags." + p + " = '" + (str(params_dict[p]) if params_dict[p] != None else "None") + "'"
        #print("[DEBUG] Filter string: " + filter_string)

        models = client.search_registered_models(filter_string=filter_string)
        already_registered_model = (len(models) > 0)

        if (already_registered_model):
            print("[DEBUG] Already registered model with name '" + model_name + "'")
            print("[DEBUG] Loading the model...")
            model_version = int(models[0].latest_versions[-1].version)
            model = pipeline.load_model(model_name=model_name, model_version=model_version)
            model_version = model_version + 1

        else:
            print("[DEBUG] Model with name '" + model_name + "' not found")
            print("[DEBUG] Training the model...")
            model_version = 1
            model = pipeline.train()

        #EVALUATION
        #When the model performance is measuered
        print("[DEBUG] Evaluating the model...")
        performance, report, ytest, ypred = pipeline.evaluate()

        #LOGGING
        #Performance metrics
        print("[DEBUG] Logging the performance metrics...")
        log_metrics(model_tags, performance)

        #Artifacts
        #We define "artifact" every contribute helping us to understand how good a model is
        print("[DEBUG] Logging the artifacts...")
        log_artifacts(performance, le, report, ytest, ypred, options["explain"])

        #REGISTER
        #When the model, if pre-tuned threshold criteria are satisfied, is stored for the future
        print("[DEBUG] Thresholds for validation criteria: " + str(thresholds_dict))

        
        #Model
        #print("[DEBUG] Logging dataset and model parameters...")        #NB: we always use mlflow.sklearn as logging module for models!
        log_model(model, model_name, model_tags, model_version, options, performance, thresholds_dict, already_registered_model, client)


        #Options and other info
        #print("[DEBUG] Logging options and other info...")
        #log_options(model_name, model_version, options, already_registered_model, client)
        
        return