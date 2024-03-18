
from sklearn.model_selection import train_test_split

def split(data, label, proportion=[0.7, 0.3, 0.0], random_state=0):

        if(len(data) == 2):         #two dataframes
            print("[DEBUG] Implicit splitting for data set.")

            df_train = data[0]
            df_test = data[1]

            # Pandas DF to list conversion
            Xtrain = df_train.iloc[:, :-2].values       #TO FIX: who said next-to-last column is the separator?
            ytrain = df_train.loc[:, label].values
            Xtest = df_test.iloc[:, :-2].values
            ytest = df_test.loc[:, label].values
        
        elif(len(data) == 1):       #one dataframe to be splitted
            print("[DEBUG] Explicit splitting (hold-out) for data set.")
            print("[DEBUG] Split proportion chosen: " + str(proportion))

            X = data[0].iloc[:, :-1].values
            y = data[0].loc[:, label].values

            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=proportion[1], train_size=proportion[0], random_state=random_state)

        return Xtrain, Xtest, ytrain, ytest
