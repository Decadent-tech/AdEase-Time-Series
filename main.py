
import pandas as pd
import numpy as np
import pylab as p
import matplotlib.pyplot as plot
from collections import Counter
import re
import os
import seaborn as sns
     

import warnings
warnings.filterwarnings("ignore")

# import the data set from google drive
train = pd.read_csv('Dataset/train_1.csv')
exog = pd.read_csv('Dataset/Exog_Campaign_eng')

# check the shape of the data
print("Shape of train data: ", train.shape)
print("Shape of exog data: ", exog.shape)
# check the first few rows of the train data
print("First few rows of train data: \n", train.head())
# check the first few rows of the exog data
print("First few rows of exog data: \n", exog.head())
# check the columns of the train data
print("Columns in train data: ", train.columns)
# check the columns of the exog data
print("Columns in exog data: ", exog.columns)
# describe the train data
print("Describe in train data: \n", train.describe())
# describe the exog data
print("Describe in exog data: \n", exog.describe())
# check the info of the train data
print("Info of train data: \n", train.info())
# check the info of the exog data
print("Info of exog data: \n", exog.info())



#missing value treatment 

print(train.isnull().sum())


# Plotting the null values in the train data
days = [r for r in range(1, len(train.columns))]
plot.figure(figsize=(10,7))
plot.xlabel('Day')
plot.ylabel('Null values')
plot.plot(days, train.isnull().sum()[1:])
plot.show()