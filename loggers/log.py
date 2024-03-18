import mlflow
import os
import json
import pickle

def log_metrics(model_tags, performance):
    
    if('accuracy' in performance.keys()):                   #'accuracy' presence ensures the other metrics are present
        print("[DEBUG] Accuracy: " + str(performance['accuracy']))
        mlflow.log_metric('accuracy', performance['accuracy'])
        model_tags['accuracy'] = performance['accuracy']

        print("[DEBUG] Average precision (macro): " + str(performance['macro_avg_precision']))
        mlflow.log_metric('macro_avg_precision', performance['macro_avg_precision'])
        model_tags['macro_avg_precision'] = performance['macro_avg_precision']

        print("[DEBUG] Average recall (macro): " + str(performance['macro_avg_recall']))
        mlflow.log_metric('macro_avg_recall', performance['macro_avg_recall'])
        model_tags['macro_avg_recall'] = performance['macro_avg_recall']

        print("[DEBUG] Average f1-score (macro): " + str(performance['macro_avg_f1-score']))
        mlflow.log_metric('macro_avg_f1-score', performance['macro_avg_f1-score'])
        model_tags['macro_avg_f1-score'] = performance['macro_avg_f1-score']

    if('AUC' in performance.keys()):
        print("[DEBUG] AUC: " + str(performance['AUC']))
        mlflow.log_metric('AUC', performance['AUC'])
        model_tags['AUC'] = performance['AUC']

        if('pAUC' in performance.keys()):
            print("[DEBUG] pAUC: " + str(performance['pAUC']))
            mlflow.log_metric('pAUC', performance['pAUC'])
            model_tags['pAUC'] = performance['pAUC']

def log_artifacts(performance, le, report, ytest, ypred, explain):

    #os.remove() disposed in order to avoid duplicate files
    #needed in case of file system log management

    if('AUC' in performance.keys()):
        print("[DEBUG] Logging ROC plot...")
        mlflow.log_artifact("roc.png")
        os.remove("roc.png")
        mlflow.log_artifact("roc.pdf")
        os.remove("roc.pdf")
    else:
        print("[DEBUG] Logging confusion matrix...")
        mlflow.log_artifact("confusion_matrix.png")
        os.remove("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.pdf")
        os.remove("confusion_matrix.pdf")

        with open("classification_report.json", "w") as outfile: 
            json.dump(report, outfile)
        mlflow.log_artifact("classification_report.json")
        os.remove("classification_report.json")

        with open("label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        mlflow.log_artifact("label_encoder.pkl")
        os.remove("label_encoder.pkl")
    
    outfile = open("ytest.txt", "w")
    for label in ytest:
        outfile.write(str(label) + '\n')
    outfile.close()
    mlflow.log_artifact("ytest.txt")
    os.remove("ytest.txt")

    outfile = open("ypred.txt", "w") 
    for pred in ypred:
        outfile.write(str(pred) + '\n')
    outfile.close()
    mlflow.log_artifact("ypred.txt")
    os.remove("ypred.txt")

    if(explain):
        print("[DEBUG] Logging explanations...")
        mlflow.log_artifact("shap_values.pkl")
        os.remove("shap_values.pkl")

        mlflow.log_artifact("shap_summary_plot.png")
        os.remove("shap_summary_plot.png")

        mlflow.log_artifact("shap_summary_plot.pdf")
        os.remove("shap_summary_plot.pdf")

    # Label Encoder logging (next...)
    # mlflow.log_artifact("")
    # os.remove("")

def log_model(model, model_name, model_tags, model_version, options, performance, thresholds, already_registered_model, client):
    
    mlflow.set_tags(model_tags)

    if(not already_registered_model):               #or different test set!

        valid = True

        for t in thresholds:
            if(performance[t] < thresholds[t]): valid = False                

        if(valid):
            print("[DEBUG] Logging dataset and model parameters (performance better than thresholds numeric contraints)...")
            mlflow.sklearn.log_model(artifact_path = model_name,
                                sk_model = model,
                                #signature = signature,
                                registered_model_name = model_name)
            
            for key in model_tags.keys():
                client.set_registered_model_tag(name = model_name, key = key, value = model_tags[key])

            log_options(model_name, model_version, options, client)
            
        else:
            print("[DEBUG] Not logging the model for insufficient performance.")




def log_options(model_name, model_version, options, client):

    #if(not already_registered_model):

    client.set_registered_model_tag(name = model_name, key = 'binarize_labels', value = options["binarize_labels"])
    client.set_registered_model_tag(name = model_name, key = 'samples_barrier', value = options["samples_barrier"])

    #Setting specific model version tag (example: no_duplicate opt.)
    client.set_model_version_tag(name = model_name, version = str(model_version), key = 'no_duplicate', value = options["no_duplicate"])