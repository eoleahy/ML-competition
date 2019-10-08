import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, StandardScaler, LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt

submission_file = "tcd ml 2019-20 income prediction submission file.csv"




WRITE = 0

def check_for_nulls(df):    
    #check for null values
    for heading in df.columns:
        col = df[heading]
        total_missing = col.isnull().sum()
        percentage_missing = (total_missing/col.size) * 100 
        print("{}:{}\n{}% missing".format(heading, total_missing, percentage_missing))


def preprocess_data(data):

    #----- Processing Age -----
    X = data["Age"].values.reshape(-1, 1)
    imputer=SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    scaler = StandardScaler().fit(X)
    data["Age"] = scaler.fit_transform(X)

    #----- Processing Height -----
    X = data["Body Height [cm]"].values.reshape(-1, 1)
    imputer=SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    scaler = StandardScaler().fit(X)
    data["Body Height [cm]"] = scaler.fit_transform(X)

    #----- Processing city size -----
    X = data["Size of City"].values.reshape(-1, 1)
    scaler = Normalizer().fit(X)
    data["Size of City"] = scaler.fit_transform(X)

    #------ Processing Income -----
    X = data["Income in EUR"].values.reshape(-1, 1)
    scaler = Normalizer().fit(X)
    data["Income in EUR"] = scaler.fit_transform(X)

    return data

def train(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #scaler = StandardScaler().fit(X_train)
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    regressor = LinearRegression()
    regressor.fit(X_train,y_train)

    #print("Correlation between age and income = {}".format(regressor.coef_))

    y_pred = cross_val_predict(regressor,X_test, y_test,cv=10)
    #y_pred = regressor.predict(X_train)

    #print(y_pred)
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    return y_pred

def multi_train(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = cross_val_predict(regressor, X_test, y_test, cv=10)
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    return y_pred

def main():

    #headings = training_data.columns
    training_data = pd.read_csv("training.csv", index_col="Instance")
    submission_data = pd.read_csv("test.csv", index_col="Instance")
    check_for_nulls(submission_data)
    processed_training_data = preprocess_data(training_data)

    #print(processed_training_data)

    #-------------Start of the actual training --------------- 
    print("Age vs income")
    X = processed_training_data["Age"].values.reshape(-1, 1)
    y = processed_training_data["Income in EUR"].values.reshape(-1, 1)
    y_pred = train(X, y)



    print("Age, Height and City vs Income")
    X = processed_training_data[["Age", "Body Height [cm]", "Size of City"]].values
    y = processed_training_data["Income in EUR"].values
    y_pred = multi_train(X, y)

    if(WRITE):
        submission_data = pd.read_csv(submission_file, index_col="Instance")
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
