import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Business Problem:
The business objective is to forecast plastic sales for the upcoming months.
Constraints: None identified.
Data Pre-processing:
Read in the data
df = pd.read_csv('PlasticSales.csv')

Check for null values
df.isnull().sum()

No null values found
Data Cleaning
Check for any unnecessary columns
df.columns

Only 'month' and 'sales' columns are necessary, so we will drop the rest
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

Feature Engineering
Convert 'month' column to datetime
df['month'] = pd.to_datetime(df['month'])

Set 'month' column as index
df.set_index('month', inplace=True)

Outlier Treatment
Check for outliers using boxplot
sns.boxplot(x=df['sales'])

Outliers are present
Use z-score method to identify and remove outliers
z = np.abs(stats.zscore(df['sales']))
threshold = 3
print(np.where(z > 3))
df = df[(z < 3)]

Exploratory Data Analysis (EDA):
Summary
df.describe()

Identify trend
Plot time series data
plt.plot(df)
plt.xlabel('Month')
plt.ylabel('Sales')
plt.title('Plastic Sales over Time')

A positive trend is observed
Identify seasonality
Use ACF and PACF plots to determine appropriate model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df)
plot_pacf(df)

Both ACF and PACF show a significant lag at 12 months, indicating seasonality
Model Building:
Perform Forecasting using data driven approach
from fbprophet import Prophet
m = Prophet()
df.reset_index(inplace=True)
df.rename(columns={'month': 'ds', 'sales': 'y'}, inplace=True)
m.fit(df)
future = m.make_future_dataframe(periods=12, freq='M')
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

Plot forecast
m.plot(forecast)
plt.title('Forecast of Plastic Sales using Data Driven Approach')

Perform Forecasting using moving averages
Calculate rolling mean and standard deviation
rolling_mean = df['y'].rolling(window=12).mean()
rolling_std = df['y'].rolling(window=12).std()

Plot rolling statistics
plt.plot(df, color='blue', label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_std, color