import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#Streamlit
st.title('Apple stock prices 30 days forecast')
start_date=st.date_input('Select Start Date')
end_date=st.date_input('Select End Date')
st.subheader('Forecasts for next 30 days')
#Import Data
df=yf.download('AAPL',start=start_date,end=end_date)
df=df.dropna()
#EDA
df1_org=df.diff()
df1=df1_org.dropna()
#Forecast
split_point=int(len(df1)*0.8)
train,test=df1[:split_point],df1[split_point:]
model = ExponentialSmoothing(train['Close'], trend=None, seasonal=None)
model_fit=model.fit()
test_predictions=model_fit.forecast(steps=len(test))
model1 = ExponentialSmoothing(df1['Close'], trend=None, seasonal=None)
model_fit1=model1.fit()
future_predicitions=model_fit1.forecast(steps=30)
future_predicitions_normal=future_predicitions.cumsum()+np.array(df['Close'].iloc[-1])
test_predictions_normal=test_predictions.cumsum()+np.array(df['Close'].iloc[split_point])
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=11), periods=30, freq='B')
future_pred=pd.DataFrame()
future_pred['Date']=forecast_dates
future_predicitions_normal=future_predicitions_normal.reset_index(drop=True)
future_pred['Close']=future_predicitions_normal
st.write("Static Table:")
st.table(future_pred)
#Visualization
st.subheader('Forecasts for next 30 days')
st.table(future_pred)
fig, ax = plt.subplots()
plt.figure(figsize=(14,5))
plt.plot(df['Close'],label='Actual',color='blue')
plt.axvline(x=train.index[-1],color='grey',linestyle='--',label='Train-Test Split')
plt.plot(forecast_dates,future_predicitions_normal,label='forecasted',color='red')
plt.plot(test.index,test_predictions_normal,label='Predicted Test',color='green')
plt.legend()
st.pyplot(fig)











