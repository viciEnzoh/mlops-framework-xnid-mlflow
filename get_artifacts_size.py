import os
import sys
import pandas as pd

def main(argv):

    #argv[1]: CSV filename to process

    #Read CSV file with experiment table
    exp_filename = argv[1]
    df = pd.read_csv(exp_filename)
    experiment_id = df.at[0, 'experiment_id']
    folder = "../mlartifacts/" + str(experiment_id)

    #Extracting run_id column
    run_id_col = df['run_id']

    size_col = []

    #Loop on run_id column for filling the artifact_size list
    for run_id in run_id_col:

        # assign size
        size = 0
        
        # assign folder path
        folderpath = folder + "/" + run_id + "/artifacts"
        
        # get size
        for path, dirs, files in os.walk(folderpath):
            for f in files:
                if("model.pkl" in f):
                    fp = os.path.join(path, f)
                    size += os.path.getsize(fp)

        size_col.append(size)

    #Attaching column to DataFrame
    df['artifact_size'] = size_col

    #Saving CSV file
    new_csv_name = exp_filename.split('.')[0] + '_size.csv'
    df.to_csv(new_csv_name, index=False, na_rep=None)

if __name__ == '__main__':
    main(sys.argv)