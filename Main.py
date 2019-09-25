import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt

DEBUG = 1

def print_all(data,columns):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(data[columns])

def main():

    training_data = pd.read_csv("training.csv",index_col="Instance")
    headings = training_data.columns
    
    if DEBUG:
        print(training_data.info())
        print(training_data)

    income = training_data["Income in EUR"]
   
    age = training_data["Age"]
    mean_income = income.mean()

    print("Mean income =", mean_income)



if __name__ == "__main__":
    main()
