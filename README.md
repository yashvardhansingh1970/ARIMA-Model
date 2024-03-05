This project presents an interactive stock price prediction dashboard, leveraging machine 
learning models such as ARIMA and LSTM. The dashboard, developed using Streamlit and 
Plotly, empowers users to forecast stock prices for any given stock ticker. The primary dataset 
is sourced from Yahoo Finance, allowing users to input their chosen stock symbol for 
predictions. After thorough evaluation, it was found that the ARIMA model consistently 
outperformed the LSTM model in terms of future price prediction accuracy. The model was 
rigorously trained and tested using five years of historical Apple Inc. (AAPL) stock price data. 
Data analysis was performed in Python, with missing data handled through z-score of logreturn standardization. 
The z-score plot aids in identifying outliers and visually inspecting 
periods of high liquidity. The dataset was split into training and testing sets, with the training 
set consisting of data from the beginning up to the last 100 data points and the testing set 
comprising the last 100 data points. To determine the best ARIMA model, 'pandarima' was 
employed to simulate a stepwise fit. Once the optimal parameters (p, d, q) were determined, 
the model was fitted to the training set, and its performance was analyzed using AIC and BIC 
values. Subsequently, the testing set was used to make predictions, providing users with a 10-
day forecast of stock prices. The dashboard also includes user-friendly input options for stock 
ticker selection, as well as start and end date customization. Dynamic plots created with Plotly 
showcase the predicted results, while 95% confidence interval plot assists users in 
comprehending the momentum range of their chosen stock. This project serves as a valuable 
tool for investors, aiding them in making informed decisions and effectively managing 
investment risks.
Link to the project: https://tableautest-mhf3a9ap8vqlrjci37nurw.streamlit.app/
