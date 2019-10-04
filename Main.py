import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

submission_file = "tcd ml 2019-20 income prediction submission file.csv"

WRITE = 0

def check_for_nulls(df):    
    #check for null values
    for heading in df.columns:
        col = df[heading]
        print("{}:{}".format(heading,col.isnull().sum()))

def fill_age(df):
    #takes in a column
    #fills in null values according to mean age
    mean_age = df.mean(skipna=True)
    return df.fillna(mean_age)

def main():

    training_data = pd.read_csv("training.csv",index_col="Instance")
    test_data = pd.read_csv("test.csv",index_col="Instance")
    #headings = training_data.columns


    #check_for_nulls(training_data)
    #check_for_nulls(test_data)

    training_data["Age"] = fill_age(training_data["Age"])
    test_data["Age"] = fill_age(test_data["Age"])

    #-------------Start of the actual training ---------------
    '''
    X = training_data["Age"].values.reshape(-1,1)
    y = training_data["Income in EUR"].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    '''
    X_train = training_data["Age"].values.reshape(-1,1)
    y_train = training_data["Income in EUR"].values.reshape(-1,1)
    X_test = test_data["Age"].values.reshape(-1,1)
    y_test = test_data["Income"].values.reshape(-1,1)

    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    print("Correlation between age and income = {}".format(regressor.coef_))

    y_pred = regressor.predict(X_test)

    #print(y_pred)
    #print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    if(WRITE):
        submission_data=pd.read_csv(submission_file,index_col="Instance")
        submission_data["Income"] = y_pred
        print(submission_data)
        submission_data.to_csv(submission_file)

  

    #---------- Multi variable regressions ----------
    '''
    X = training_data[["Age","Size of City","Body Height [cm]"]].values
    y = training_data["Income in EUR"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    print(regressor.coef_)
    y_pred = regressor.predict(X_test)

    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    '''
if __name__ == "__main__":
    main()
