import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn import metrics


submission_file = "tcd ml 2019-20 income prediction submission file.csv"


#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)


WRITE = 0 #Set to 1 to write to submission file

def check_for_nulls(df):    

    """
    Checks a given 2d dataframe for null values and prints
    the name of the column with the amount of nulls. 
    """
    #check for null values
    for heading in df.columns:
        col = df[heading]
        total_missing = col.isnull().sum()
        print("{}:{}".format(heading, total_missing))

def dummy_encode(data, col):

    """
    One hot label encodes the data by adding new columns
    and dropping the old one. By deafult it drops dummy variables

    @param data: the dataframe containing the column you wish to encode

    @param col: a string name of the column you wish to encode


    returns - the dummy encoded dataframe
    """
    print("Dummy encoding {} ...".format(col))
    pre=col[0:3]
    encoded = pd.get_dummies(data[col], prefix = pre, drop_first = True)
    data.drop(col, axis=1, inplace = True)
    data = pd.concat([data,encoded], axis=1)
    return data

def preprocess_data(data):

    """
    Function to process the data, getting rid of nulls,
    imputing values and encoding low categorical data.

    @param data: the dataframe you want to process

    Returns the dataframe with cleaned up data
    """
    
    print("Processing inputs...")

    #dummy_encode(data, "University Degree")
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

    data = dummy_encode(data, "University Degree")

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
   
    data = dummy_encode(data, "Gender")


    #----- Processing Age -----
    X = data["Age"].values.reshape(-1, 1)
    age_imputer = SimpleImputer(strategy="median")
    data["Age"] = age_imputer.fit_transform(X)

    #----- Processing Profession & Country ----- 
    #Most frequent takes too long, forward fill instead
    data["Profession"] = data["Profession"].fillna("Ffill")
    data["Country"] = data["Country"].fillna("Ffill")

    #----- Dropping low correlation data -----
    data.drop("Wears Glasses", axis=1, inplace = True)
    #data.drop("Body Height [cm]", axis=1, inplace = True)
    data.drop("Hair Color", axis=1, inplace = True)

    return data
    

def targEncode(training, test, col, target):

    """
    Target encodes the training and submission data
    by mapping mean with columns
    Fills leftover NaNs with median.

    It target encodes the training and submission data,
    probably better to change the function to do it seperately.

    @param training: training data dataframe

    @param test: submission data datafram

    @param col: string representation of the column to
    target encode

    @param target: the target column (income)


    Returns a tuple containing the encoded dataframes
    """

    print("Encoding {} ...".format(col))
    
    median_income = training[target].median()
    mean_income = training.groupby(col)[target].mean()

    training[col] = training[col].map(mean_income)
    training[col] = training[col].fillna(value = median_income)

    test[col] = test[col].map(mean_income)
    test[col] = test[col].fillna(value = median_income)

    return(training, test)

def train(X, y, X_submit):

    """The function that does all the fitting and predictions.
        Uses random forest regression
    
    @param X : training data values

    @param y : target column (income) values

    @param X_submit: submission data values

    Returns the predicted income
    """

    #input("Continue?")
    print("Training model...")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaler = StandardScaler()
    X_submit = scaler.fit_transform(X_submit)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = RandomForestRegressor(n_estimators=200, max_depth= 12, random_state = 0)
    regressor.fit(X_train, y_train)

    print("Running predictions...")
    y_pred = regressor.predict(X_test)
    print("Test:")
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    y_pred1 = regressor.predict(X_submit)

    return y_pred1
  

def main():

    training_data = pd.read_csv("training.csv", index_col="Instance")
    submission_data = pd.read_csv("test.csv", index_col="Instance")
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
