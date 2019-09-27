import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import matplotlib.pyplot as plt

DEBUG = 0


def check_for_nulls(df):    
    #check for null values

    has_null = False
    for heading in df.columns:
        col = df[heading]
        print("{}:{}".format(heading,col.isnull().sum()))

        if(col.isnull().values.any() == True):
            has_null = True
      
    return has_null

def fill_age(df):
    #takes in a column
    #fills in null values according to mean age
    mean_age = df.mean(skipna=True)
    return df.fillna(mean_age)

def main():

    training_data = pd.read_csv("training.csv",index_col="Instance")
    test_data = pd.read_csv("test.csv",index_col="Instance")
    headings = training_data.columns
    
    if DEBUG:
        print("Headings={}".format(headings))
        print("{}".format(training_data.head(50)))
        print(training_data.describe())

    nulls= check_for_nulls(training_data)

    training_data["Age"] = fill_age(training_data["Age"])

    X = training_data["Age"].values.reshape(-1,1)
    y = training_data["Income in EUR"].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    print("Correlation between age and income = {}".format(regressor.coef_))

    y_pred = regressor.predict(X_test)

    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    #---------- Multi variable regressions ----------
    X = training_data[["Age","Size of City","Body Height [cm]"]].values
    y = training_data["Income in EUR"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    print(regressor.coef_)
    y_pred = regressor.predict(X_test)

    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

if __name__ == "__main__":
    main()
