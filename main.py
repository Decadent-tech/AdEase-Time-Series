
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

# From the above plot it is evident that the no. of Null values reduce overtime. We can say that initially these pages were not created so they don't have any views.

# We are dropping the rows having all NULL values first and then dropping the rows having more than 300 values as NULL becuase we have a total of 551 records, if we have more than 300 NULLs it means half of our data is NULL and that would not help in model creation.


train=train.dropna(how='all')
#‘all’ : If all values are NA, drop that row or column.
print(train.shape)
#‘thresh’ : Require that many non-NA values.
train=train.dropna(thresh=300)
print(train.shape)


# Filling all the remaining NULLs with 0


train=train.fillna(0)
     

days = [r for r in range(1, len(train.columns))]
plot.figure(figsize=(10,7))
plot.xlabel('Day')
plot.ylabel('Null values')
plot.plot(days, train.isnull().sum()[1:])
plot.show()


# EDA

#split the page column into 4 columns - page, language, region, and type
def split_page(page):
  w = re.split('_|\.', page)
  return ' '.join(w[:-5]), w[-5], w[-2], w[-1]

li = list(train.Page.apply(lambda x: split_page(str(x))))
df = pd.DataFrame(li, columns=['Title', 'Language','Access_type','Access_origin'])
df = pd.concat([train, df], axis = 1)
print (df.head())



sns.countplot(df["Language"])
plot.show()
sns.countplot(df["Access_type"])
plot.show()
sns.countplot(df["Access_origin"])
plot.show()


#  We have found lots of records for "commons". Let's check it and see if they can be further splitted into any language or not

print(df[df["Language"] == "commons"])
print(df[df["Language"] == "www"])

for i in df[(df["Language"] == "commons") | (df["Language"] == "www")].index:
    df.loc[i, "Language"] = "no_lang"

# Now we have 7 languages and 1 no_lang  - en, no_lang, de, zh , fr , ru , de ,ja , es 
sns.countplot(df["Language"])
plot.show()
# Now we can group the data by language and get the mean of the views for each language
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_lang = df.groupby("Language")[numeric_cols].mean().transpose()
df_lang.reset_index(inplace=True)
df_lang.set_index('index', inplace=True)
print(df_lang.head(10))

# Plotting the views per page for each language
df_lang.plot(figsize=(12,7))
plot.ylabel('Views per Page')
plot.show()     

''' 
# Insights from the above plotq
Looking at the above plot we can say that language - English is preferred over others. 
People view pages in English language way more than the rest.
Also there is a very interesting insight that there are some peaks in the data, especially if 
we look at 2016-08-04 in both en and ru language. 
To study about this in detail, we will look at the Exogenous data provided to us for en language.

'''


# Studying the patterns for Language - "en"


df_en = df_lang["en"]
df_en = df_lang["en"].to_frame().reset_index()
# Rename columns
df_en.columns = ["date", "views"]
df_en.set_index('date', inplace=True)
print(df_en.head())

# linear interpolation
df_en.views = df_en.views.interpolate(method='linear')

# anomalies - clip quantiles
df_en.views = df_en.views.clip(upper=df_en.views.quantile(0.98), lower=df_en.views.quantile(0.02))

# plot
df_en.views.plot(style='-o', figsize=(20,6))

# Plotting the ACF and PACF to check for seasonality and trend
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df_en.views, lags=50) # autocorrelation function
plot.show()
plot_pacf(df_en.views, lags=50)# partial autocorrelation function
plot.show()


# Let's check if our TS is stationary or not with the help of Dickey Fuller Test


import statsmodels.api as sm
def adf_test(dataset):
   pvalue = sm.tsa.stattools.adfuller(dataset)[1]
   if pvalue <= 0.05:
      print('Sequence is stationary')
   else:
      print('Sequence is not stationary')

adf_test(df_en.views)
     
#Sequence is not stationary

#As the TS is not stationary let's use differencing to make it stationary
df_en_st = df_en.copy()
df_en_st.views = df_en_st.views.diff(1)
df_en_st.dropna(inplace=True)
adf_test(df_en_st.views) # dickey fuller test again
#Sequence is stationary

# Differencing has worked here. The TS is now stationary and can be used for forecasting.

plot_acf(df_en_st.views);
plot.show()
plot_pacf(df_en_st.views);
plot.show()

# Now we can use SARIMAX model for forecasting

import statsmodels.api as sm
train=df_en[:520] # Use the previous  520 days for testing
test=df_en[520:] # Use the last 35 days for testing
model=sm.tsa.statespace.SARIMAX(train,order=(4, 1, 3))
results=model.fit()
fc=results.forecast(30,dynamic=True)
# Make as pandas series
fc_series = pd.Series(fc)
# Plot
train.index=train.index.astype('datetime64[ns]')
test.index=test.index.astype('datetime64[ns]')
plot.figure(figsize=(12,5), dpi=100)
plot.plot(train, label='training')
plot.plot(test, label='actual')
plot.plot(fc_series, label='forecast')
plot.title('Forecast vs Actuals')
plot.legend(loc='upper left', fontsize=8)
plot.show()

# metrics to check the accuracy of the model
mape = np.mean(np.abs(fc - test.views)/np.abs(test.views))
rmse = np.mean((fc - test.views)**2)**.5
print("mape:",mape)
print("rsme:",rmse)

# Using exog in SARIMAX 

ex=exog['Exog'].to_numpy()

train=df_en[:520]
test=df_en[520:]
model=sm.tsa.statespace.SARIMAX(train,order=(4, 1, 3),seasonal_order=(1,1,1,7),exog=ex[:520])
results=model.fit()

fc=results.forecast(30,dynamic=True,exog=pd.DataFrame(ex[520:]))

# Make as pandas series
fc_series = pd.Series(fc)
# Plot
train.index=train.index.astype('datetime64[ns]')
test.index=test.index.astype('datetime64[ns]')
plot.figure(figsize=(12,5), dpi=100)
plot.plot(train, label='training')
plot.plot(test, label='actual')
plot.plot(fc_series, label='forecast')
plot.title('Forecast vs Actuals')
plot.legend(loc='upper left', fontsize=8)
plot.show()

# metrics to check the accuracy of the model
from sklearn.metrics import (
    mean_squared_error as mse,
    mean_absolute_error as mae,
    mean_absolute_percentage_error as mape
)

# Creating a function to print values of all these metrics.
def performance(actual, predicted):
    print('MAE :', round(mae(actual, predicted), 3))
    print('RMSE :', round(mse(actual, predicted)**0.5, 3))
    print('MAPE:', round(mape(actual, predicted), 3))

mape = np.mean(np.abs(fc - test.views)/np.abs(test.views))
rmse = np.mean((fc - test.views)**2)**.5
print("mape:",mape)
print("rsme:",rmse)

# FB Prophet 
