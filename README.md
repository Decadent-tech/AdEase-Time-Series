Ad Ease is an ads and marketing based company helping businesses elicit maximum clicks @ minimum cost. AdEase is an ad infrastructure to help businesses promote themselves easily, effectively, and economically. The interplay of 3 AI modules - Design, Dispense, and Decipher, come together to make it this an end-to-end 3 step process digital advertising solution for all.You are working in the Data Science team of Ad ease trying to understand the per page view report for different wikipedia pages for 550 days, and forecasting the number of views so that you can predict and optimize the ad placement for your clients. You are provided with the data of 145k wikipedia pages and daily view count for each of them. Your clients belong to different regions and need data on how their ads will perform on pages in different languages.

## Data Dictionary:

## 📂 Dataset
## Dataset:  https://drive.google.com/drive/folders/1mdgQscjqnCtdg7LGItomyK0abN6lcHBb
Two datasets are used:

1. **train_1.csv** — Wikipedia page views dataset containing:
   - `Page`: Page name including title, language, access type, and origin.
   - Daily view counts for multiple pages over time.

2. **Exog_Campaign_eng** — Exogenous data representing external campaign activity that may influence views.



# Wikipedia Page Views Forecasting

## 📌 Project Overview
This project performs **time series analysis and forecasting** of Wikipedia page views using different statistical and machine learning approaches.  
It explores the impact of **language, access type, and access origin** on page views and applies **SARIMAX** and **Facebook Prophet** (with and without exogenous variables) to predict future trends.

Key steps include:
- Data cleaning and missing value treatment.
- Exploratory Data Analysis (EDA) with language-based trends.
- Time series decomposition and stationarity checks.
- Forecasting using SARIMAX (with/without exogenous variables).
- Forecasting using Facebook Prophet (with/without exogenous variables).
- Model evaluation using **MAPE, RMSE, and MAE**.

## ⚙️ Installation & Setup
### 1️⃣ Clone the repository

git clone https://github.com/your-username/wiki-pageviews-forecasting.git
cd wiki-pageviews-forecasting


### 2️⃣ Install dependencies
    pip install -r requirements.txt

### requirements.txt should include:
pandas
numpy
matplotlib
seaborn
statsmodels
prophet
scikit-learn

### 📊 Exploratory Data Analysis

    Split the Page column into Title, Language, Access_type, and Access_origin.
### Handle missing values:
        Drop rows with all values missing.
        Drop rows with >300 missing values.
        Fill remaining missing values with 0.
## Analyze and visualize:

Language distribution.
Access type and origin.
Trends in average daily views by language.

### 🔍 Time Series Analysis

Focus on English (en) page views for forecasting.
Interpolate missing values and clip outliers.
Use ADF Test to check stationarity.
Apply differencing to make series stationary.
Examine ACF and PACF plots for ARIMA order selection.

### 📈 Forecasting Models
1. SARIMAX
Without exogenous variables.
With exogenous variables (Exog_Campaign_eng).
Seasonal order set as (1,1,1,7) for weekly seasonality.

2. Facebook Prophet
Without exogenous variables.
With exogenous variables as regressors.

### 📏 Evaluation Metrics
The following metrics are computed:
MAE – Mean Absolute Error
RMSE – Root Mean Squared Error
MAPE – Mean Absolute Percentage Error

Example:
mape: 0.05
rsme: 161.59

### 📌 Key Insights

English pages dominate traffic compared to other languages.
Significant traffic peaks on specific dates (e.g., 2016-08-04 for en and ru).
SARIMAX with exogenous variables generally outperforms FB Prophet in this dataset.
The time series has both trend and seasonality components.
Differencing order of 1 is required for stationarity.

### 🚀 How to Run
python main.py

### The script will:

Perform data cleaning and EDA.
Run SARIMAX and Prophet models.
Plot forecasts vs. actuals.
Print evaluation metrics.

### 📜 Conclusion

SARIMAX is the best-performing model for this dataset.
External campaign data improves forecasting accuracy.
Prophet is good at capturing seasonality but underperforms compared to SARIMAX in this case.

### 📎 References

Statsmodels Documentation  
Facebook Prophet Documentation
Wikipedia Pageviews Dataset
