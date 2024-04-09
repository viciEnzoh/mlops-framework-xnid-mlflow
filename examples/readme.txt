USAGE GUIDELINES
First, you can choose the desired JSON file to reproduce the example, and then copy/paste its content to replace the contents of 'exp_setup.json' in the up folder.
Next, you need to launch the script via the console: 'python3 main_exps.py'
Remember to start from the 'intrusion-detection/' folder.
You can examine the results using the MLflow GUI. To achieve this, you can visualize it in a browser by entering '<tracking_server_ip>:5000'.


EXAMPLES
> 'ad_alad_paper.json': optimization of ALAD model for anomaly detection task, as reported in the paper
> 'bmd_paper.json': optimization of XGBoost model for binary misuse detection task, as reported in the paper
> 'mmd_paper.json': optimization of Random Forest model for multiclass misuse detection task, as reported in the paper
