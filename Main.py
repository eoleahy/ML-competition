import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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

training_income=""

WRITE = 0

def check_for_nulls(df):    
    #check for null values
    for heading in df.columns:
        col = df[heading]
        total_missing = col.isnull().sum()
        #percentage_missing = (total_missing/col.size) * 100 
        print("{}:{}".format(heading, total_missing))


def preprocess_data(data):
    
    

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


    #uni_labelencoder = LabelEncoder()
    #data["University Degree"] = uni_labelencoder.fit_transform(data["University Degree"])

    encoded_degree = pd.get_dummies(data["University Degree"], prefix='degree') 
    data.drop("University Degree", axis=1, inplace = True)
    data = pd.concat([data,encoded_degree], axis=1)
    
    #----- Processing Hair-----
    data.drop("Hair Color", axis=1, inplace = True)
    '''
    X = data["Hair Color"].values.reshape(-1 ,1)
    hair_imputer = SimpleImputer(strategy="constant", fill_value="Other")#Imputer for empty cells
    X = hair_imputer.fit_transform(X)
    hair_imputer = SimpleImputer(missing_values="0",
                                strategy="constant", 
                                fill_value="Other")#Imputer for 0 cells

    X = hair_imputer.fit_transform(X)
    hair_imputer = SimpleImputer(missing_values="Unknown",
                                strategy="constant", 
                                fill_value="Other")#Imputer for "Unknown" cells

    X = hair_imputer.fit_transform(X)
    data["Hair Color"] = X    
    
    encoded_hair = pd.get_dummies(data["Hair Color"], prefix='hair', drop_first=True) 
    data.drop("Hair Color", axis=1, inplace = True)
    data = pd.concat([data,encoded_hair], axis=1)
    '''
    #----- Processing Gender -----
    X = data["Gender"].values.reshape(-1, 1)
    gender_imputer = SimpleImputer(strategy="most_frequent")#Imputer for empty cells

    X = gender_imputer.fit_transform(X)

    gender_imputer = SimpleImputer(missing_values="unknown", 
                                    strategy="most_frequent")#Imputer for "unknown cells"

    X = gender_imputer.fit_transform(X)

    gender_imputer = SimpleImputer(missing_values="0", 
                                    strategy="most_frequent")#Imputer for 0 cell

    X = gender_imputer.fit_transform(X)
    data["Gender"] = X
   
    #print(data["Gender"].unique())
    encoded_gender = pd.get_dummies(data["Gender"], prefix='gender',drop_first=True) 
    data.drop("Gender", axis=1, inplace = True)
    data = pd.concat([data,encoded_gender], axis=1)


    #----- Processing Age -----
    X = data["Age"].values.reshape(-1, 1)
    age_imputer = SimpleImputer(strategy="median")
    data["Age"] = age_imputer.fit_transform(X)


    #----- Processing Height -----
    X = data["Body Height [cm]"].values.reshape(-1, 1)
    height_imputer=SimpleImputer(strategy="mean")
    data["Body Height [cm]"] = height_imputer.fit_transform(X)


    #----- Processing Profession & Country ----- 
    #Most frequent takes too long, forward fill instead
    data["Profession"] = data["Profession"].fillna("Ffill")
    data["Country"] = data["Country"].fillna("Ffill")

    #print(data)

    data.drop("Wears Glasses", axis=1, inplace = True)

    return data

def targEncode(training, test, target):

    #Target encoding profession and country
   # Income
    
    training1 = training.drop("Income in EUR", axis = 1)

    ce_target = ce.TargetEncoder(return_df=True)
    ce_target.fit(training1, target)
    training1 = ce_target.transform(training1, target)
    training = pd.concat([training1, target], axis =1)
    
    test1 = test.drop("Income", axis = 1)

    test1 = ce_target.transform(test1, target[:73230])
    test = pd.concat([test1, test["Income"]], axis = 1)

    #print(training)
    
    #training.drop("Profession", axis= 1, inplace=True)
    #test.drop("Profession",axis= 1, inplace=True)

    #training.drop("Country", axis= 1, inplace=True)
    #test.drop("Country",axis= 1, inplace=True)

    scaler = StandardScaler()
    training["Profession"] = scaler.fit_transform(training["Profession"].values.reshape(-1,1))
    test["Profession"] = scaler.transform(test["Profession"].values.reshape(-1,1))

    scaler = StandardScaler()
    training["Age"] = scaler.fit_transform(training["Age"].values.reshape(-1,1))
    test["Age"] = scaler.transform(test["Age"].values.reshape(-1,1))

    scaler = StandardScaler()
    training["Country"] = scaler.fit_transform(training["Country"].values.reshape(-1,1))
    test["Country"] = scaler.transform(test["Country"].values.reshape(-1, 1))

    scaler = StandardScaler()
    training["Size of City"] = scaler.fit_transform(training["Size of City"].values.reshape(-1,1))
    test["Size of City"] = scaler.transform(test["Size of City"].values.reshape(-1,1))

    scaler = StandardScaler()
    training["Year of Record"] = scaler.fit_transform(training["Year of Record"].values.reshape(-1,1))
    test["Year of Record"] = scaler.transform(test["Year of Record"].values.reshape(-1, 1))

    scaler = StandardScaler()
    training["Body Height [cm]"] = scaler.fit_transform(training["Body Height [cm]"].values.reshape(-1,1))
    test["Body Height [cm]"] = scaler.transform(test["Body Height [cm]"].values.reshape(-1,1))
    
    print(training)
    print(test)
    return(training, test)

def multi_train(X, y, X_submit):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    
    y_pred = cross_val_predict(regressor, X_test, y_test,cv=10)
    #y_pred = regressor.predict(X_test)
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    


    #scaler = StandardScaler()
    #X_submit1 = scaler.fit_transform(X_submit)

    y_pred1 = regressor.predict(X_submit)

    print("Submission test: ")
    print('Mean Squared Error:', metrics.mean_squared_error(y[:73230], y_pred1))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y[:73230], y_pred1)))
    
    return y_pred1
  

def main():

    #headings = training_data.columns
    training_data = pd.read_csv("training.csv", index_col="Instance")
    submission_data = pd.read_csv("test.csv", index_col="Instance")
    #check_for_nulls(submission_data)
    processed_training_data = preprocess_data(training_data)
    processed_submission_data = preprocess_data(submission_data)

    processed_training_data, processed_submission_data = targEncode(
                                                        processed_training_data, 
                                                        processed_submission_data, 
                                                        processed_training_data["Income in EUR"])

    #check_for_nulls(processed_submission_data)
    #print(processed_submission_data)

    #-------------Start of the actual training --------------- 

    y = processed_training_data["Income in EUR"].values
    #print(processed_training_data.drop("Income in EUR", axis = 1).columns)
    X = (processed_training_data.drop("Income in EUR", axis = 1)).values
    X_submit = (processed_submission_data.drop("Income", axis = 1))

    #print(X_submit)
    y_pred = multi_train(X, y, X_submit.values)


    if(WRITE):
        submission_data = pd.read_csv(submission_file, index_col="Instance")
        submission_data["Income"] = y_pred
        #print(submission_data)
        submission_data.to_csv(submission_file)

    

if __name__ == "__main__":
    main()
