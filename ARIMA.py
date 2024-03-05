import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go


# start_date = '2018-08-10'
# end_date = '2023-09-06'

st.title('Stock Trend Prediction')
#user_input
user_input = st.text_input('Enter Stock Ticker','AAPL')
start_date = st.date_input("Select a start date:")

# Create a date input field for the user to select an end date
end_date = st.date_input("Select an end date:")

if start_date <= end_date:
    # Fetch historical stock data using yfinance
    try:
        # Download the historical data
        df = yf.download(user_input, start=start_date, end=end_date)

        # Display the historical data
        st.write(f"Historical data for {user_input} from {start_date} to {end_date}:")
        st.write(df)

    except Exception as e:
        st.error(f"Error fetching data: {e}")
else:
    st.error("End date must be greater than or equal to start date.")
# df = yf.download(user_input, start=start_date, end=end_date)

st.subheader('Closing Price vs Time Chart')
trace = go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Closing Price')

layout = go.Layout(
    title=f"{user_input} Closing Price",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Price (USD)")
)

fig = go.Figure(data=[trace], layout=layout)

# Show the interactive plot in the Jupyter Notebook
st.plotly_chart(fig)
fig, ax = plt.subplots(figsize=(20, 8))
sns.set_style('whitegrid')


st.subheader('Volume Traded')
trace = go.Bar(x=df.index, y=df["Volume"], name='Volume Traded', marker_color='blue')
layout = go.Layout(
    title=f"{user_input} Volume Traded",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Volume")
)

fig = go.Figure(data=[trace], layout=layout)
st.plotly_chart(fig)

#Z-score plot
st.subheader('Z-Score of log return Standardization')

# calculate the log-returns
df['Log-Return'] = np.log(df['Close'] / df['Close'].shift(1))

# calculate the mean and variance
mean = np.mean(df['Log-Return'])
variance = np.var(df['Log-Return'])
std_dev = np.sqrt(variance)

# normalize to z-score
df['Z-Score'] = (df['Log-Return'] - mean) / std_dev

first_close_price = df.iloc[0]['Close']
last_close_price = df.iloc[-1]['Close']
percentage_increase = (last_close_price) / first_close_price * 100

#metric information
st.subheader('Mean, Variance, Std Deviation & Percentage Increase')
# st.write(df.head())
st.write("Mean = " + str(mean))
st.write("Variance = " +str(variance))
st.write("Std Deviation = " +str(std_dev))
st.write("Percentage Increase = "+str(percentage_increase)+"%")


df.reset_index(inplace=True)
df.head()
df.Date= pd.to_datetime(df.Date)
df2 =df.set_index('Date')
df2.head()
data = list(df2["Close"])

# from pmdarima.arima.utils import ndiffs
# d_value = ndiffs(data,test = "adf")
# print("d value:", d_value)

from pmdarima.arima.utils import ndiffs
d_value = ndiffs(data,test = "adf")
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMAResults
x_train= data[:-100]
x_test = data[-100:]
print(len(x_train),len(x_test))
stepwise_fit = auto_arima(data,trace=True,suppress_warnings=True)

import statsmodels.api as sm
model = sm.tsa.arima.ARIMA(data, order=(0,1,1))
model = model.fit()





# model.summary()

start=len(x_train)
end=len(x_train)+len(x_test)-1
pred = model.predict(start=start,end=end)
# pred
s = pd.Series(pred, index =df2.index[-100:])


st.subheader('Actual Stock Price vs Predicted Price')
trace_actual = go.Scatter(x=df2.index[-100:], y=df2['Close'][-100:], mode='lines', name='Actual Stock Price')
# Create a trace for the predicted price
trace_predicted = go.Scatter(x=df2.index[-100:], y=s[-100:], mode='lines', name='Predicted Price')

# Create the layout for the plot
layout = go.Layout(
    xaxis=dict(title='Date'),
    yaxis=dict(title='Price'),legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    ))

# Create a figure with the traces and layout
fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
# fig= plt.figure(figsize=(20,8), dpi=100)
st.plotly_chart(fig)

# Show the plot
# fig.show()


# st.subheader('Actual Stock Price vs Predicted Price')
# s = pd.Series(pred, index =df2.index[-100:])
# # s
# fig2= plt.figure(figsize=(8,4), dpi=100)
# df2['Close'][-100:].plot(label='Actual Stock Price', legend=True)
# s.plot(label='Predicted Price', legend=True)
# st.pyplot(fig2)

from statsmodels.graphics.tsaplots import plot_predict


st.subheader('95% Confidence Interval')
fig3= plt.figure(figsize=(8,4), dpi=100)
fig3= plot_predict(model, start = len(data)-500, end = len(data)+10, dynamic = False);
st.pyplot(fig3)


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(x_test,pred))

from sklearn.metrics import r2_score
r2_score(x_test,pred)



#Future Predictions
pred_future = model.predict(start=end,end=end+10)
# pred_future

import datetime
start_date = datetime.datetime(2024,3,4)
dates = [start_date + datetime.timedelta(days=idx) for idx in range(11)]

pred_future2 = pd.Series(pred_future, index = dates)
# pred_future2

st.subheader('Actual Stock Price vs Future Predicted Price')
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=10,4
fig= plt.figure(dpi=100)
df2['Close'][-300:].plot(label='Actual Stock Price', legend=True)
pred_future2.plot(label='Future Predicted Price', legend=True)
st.pyplot(fig)



start_date = '2024-03-04'
end_date = '2024-03-15'

check_df = yf.download(user_input, start=start_date, end=end_date)
check_df.reset_index(inplace=True)
check_df.head()

check_df.Date= pd.to_datetime(check_df.Date)
check_df2 =check_df.set_index('Date')

# st.subheader('Future Predicted Price vs Actual Stock Price')
# fig2 = plt.figure(figsize=(9,5), dpi=100)
# plt.subplot(1, 2, 1)
# check_df2['Close'].plot(label='Actual Stock Price', legend=True)
# plt.subplot(1, 2, 2)
# pred_future2.plot(label='Future Predicted Price', legend=True, color='orange')
# st.pyplot(fig2)

from plotly.subplots import make_subplots
st.subheader('Future Predicted Price vs Actual Stock Price')
fig = make_subplots(rows=1, cols=2, subplot_titles=("Future Predicted Price", "Actual Stock Price"))


# Add the trace for Actual Stock Price to the second subplot
trace_actual = go.Scatter(x=check_df2.index, y=check_df2['Close'], mode='lines', name='Actual Stock Price')
fig.add_trace(trace_actual, row=1, col=2)

# Add the trace for Future Predicted Price to the first subplot
trace_predicted = go.Scatter(x=pred_future2.index, y=pred_future2, mode='lines', name='Future Predicted Price', line=dict(color='red'))
fig.add_trace(trace_predicted, row=1, col=1)

# Update the layout of the figure
fig.update_layout(showlegend=True)
st.plotly_chart(fig)
# # Show the plot
# fig.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse= np.sqrt(mean_squared_error(x_test,pred))
mae= np.sqrt(mean_absolute_error(x_test,pred))
st.write("Mean Squared error= " +str(mse))
st.write("Mean Absolute error = "+str(mae))
