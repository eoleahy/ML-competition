import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce

submission_file = "tcd ml 2019-20 income prediction submission file.csv"


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)


WRITE = 0

def check_for_nulls(df):    
    #check for null values
    for heading in df.columns:
        col = df[heading]
        total_missing = col.isnull().sum()
        #percentage_missing = (total_missing/col.size) * 100 
        print("{}:{}".format(heading, total_missing))


def preprocess_data(data):
    
    print("Processing inputs...")

    #data = data.fillna(method="ffill")

    #----- Processing Year -----
    X = data["Year of Record"].values.reshape(-1, 1)
    year_imputer = SimpleImputer(strategy="mean")
    data["Year of Record"] = year_imputer.fit_transform(X)


    #----- Processing Univeristy Degree -----
    X = data["University Degree"].values.reshape(-1, 1)
    uni_imputer = SimpleImputer(strategy="constant", fill_value="No") #Imputer for empty cells
    X = uni_imputer.fit_transform(X)
    uni_imputer = SimpleImputer(missing_values="0",
                                strategy="constant",  
                                fill_value="No")#Imputer for 0 cells
                                
    data["University Degree"] = uni_imputer.fit_transform(X)

    encoded_degree = pd.get_dummies(data["University Degree"], prefix='degree', drop_first=True) 
    data.drop("University Degree", axis=1, inplace = True)
    data = pd.concat([data,encoded_degree], axis=1)

    #----- Processing Gender -----
    X = data["Gender"].values.reshape(-1, 1)
    gender_imputer = SimpleImputer(strategy="most_frequent")#Imputer for empty cells

    X = gender_imputer.fit_transform(X)

    gender_imputer = SimpleImputer(missing_values="unknown", 
                                    strategy="most_frequent")#Imputer for unknown cells

    X = gender_imputer.fit_transform(X)

    gender_imputer = SimpleImputer(missing_values="0", 
                                    strategy="most_frequent")#Imputer for 0 cell

    X = gender_imputer.fit_transform(X)
    data["Gender"] = X
   
    encoded_gender = pd.get_dummies(data["Gender"], prefix='gender',drop_first=True) 
    data.drop("Gender", axis=1, inplace = True)
    data = pd.concat([data,encoded_gender], axis=1)


    #----- Processing Age -----
    X = data["Age"].values.reshape(-1, 1)
    age_imputer = SimpleImputer(strategy="median")
    data["Age"] = age_imputer.fit_transform(X)

    #----- Processing Profession & Country ----- 
    #Most frequent takes too long, forward fill instead
    data["Profession"] = data["Profession"].fillna("Ffill")
    data["Country"] = data["Country"].fillna("Ffill")

    #----- Processing Hair -----
    '''
    X = data["Hair Color"].values.reshape(-1, 1)
    hair_imputer = SimpleImputer(strategy="constant", fill_value="Other") #Imputer for empty cells
    X = hair_imputer.fit_transform(X)
    hair_imputer = SimpleImputer(missing_values="0",
                                strategy="constant",  
                                fill_value="Other")#Imputer for 0 cells
    X = hair_imputer.fit_transform(X)
    hair_imputer = SimpleImputer(missing_values="Unknown",
                                strategy="constant",  
                                fill_value="Other")#Imputer for 0 cells
    X = hair_imputer.fit_transform(X)

    data["Hair Color"] = X

    encoded_hair = pd.get_dummies(data["Hair Color"], prefix='hair',drop_first = True) 
    data.drop("Hair Color", axis=1, inplace = True)
    data = pd.concat([data,encoded_hair], axis=1)

    '''


    #----- Dropping low correlation data -----
    data.drop("Wears Glasses", axis=1, inplace = True)
    #data.drop("Body Height [cm]", axis=1, inplace = True)
    data.drop("Hair Color", axis=1, inplace = True)

    return data
    

def targEncode(training, test, col, target):

    print("Encoding {} ...".format(col))
    
    median_income = training[target].median()
    mean_income = training.groupby(col)[target].mean()

    training[col] = training[col].map(mean_income)
    training[col] = training[col].fillna(value = median_income)

    test[col] = test[col].map(mean_income)
    test[col] = test[col].fillna(value = median_income)

    return(training, test)

def train(X, y, X_submit):

    #input("Continue?")
    print("Training model...")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaler = StandardScaler()
    X_submit = scaler.fit_transform(X_submit)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = RandomForestRegressor(n_estimators=20, random_state = 0)
    regressor.fit(X_train, y_train)

    print("Running predictions...")
    y_pred = regressor.predict(X_test)
    print("Test:")
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    print(y_pred.mean())

    #for i in y_pred:
        #print(i)

    y_pred1 = regressor.predict(X_submit)

    print(y_pred1.mean())

    #for i in y_pred1:
        #print(i)

    return y_pred1
  

def main():

    training_data = pd.read_csv("training.csv", index_col="Instance")
    submission_data = pd.read_csv("test.csv", index_col="Instance")
    #check_for_nulls(submission_data)
    processed_training_data = preprocess_data(training_data)
    processed_submission_data = preprocess_data(submission_data)


    processed_training_data, processed_submission_data = targEncode(
                                                        processed_training_data, 
                                                        processed_submission_data,
                                                        "Profession", 
                                                        "Income in EUR")


    processed_training_data, processed_submission_data = targEncode(
                                                        processed_training_data, 
                                                        processed_submission_data,
                                                        "Country", 
                                                        "Income in EUR")                                                    

    #check_for_nulls(processed_submission_data)
    #print(processed_training_data)
    #print(processed_submission_data)

    #-------------Start of the actual training --------------- 

    y = processed_training_data["Income in EUR"].values
    X = (processed_training_data.drop("Income in EUR", axis = 1)).values
    X_submit = (processed_submission_data.drop("Income", axis = 1)).values
    y_pred = train(X, y, X_submit)


    if(WRITE):
        print("Writing to file...")
        submission_data = pd.read_csv(submission_file, index_col="Instance")
        submission_data["Income"] = y_pred
        #print(submission_data)
        submission_data.to_csv(submission_file)

    

if __name__ == "__main__":
    main()
