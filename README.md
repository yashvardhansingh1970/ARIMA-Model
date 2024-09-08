# Assessing-Financial-Market-Assets-Value using Time Series Forecasting Model 
This project presents an interactive stock price prediction dashboard, leveraging machine 
learning models such as ARIMA and LSTM. The dashboard, developed using Streamlit and 
Plotly, empowers users to forecast stock prices for any given stock ticker. The primary dataset 
is sourced from Yahoo Finance, allowing users to input their chosen stock symbol for 
predictions. After thorough evaluation, it was found that the ARIMA model consistently 
outperformed the LSTM model in terms of future price prediction accuracy. The model was 
rigorously trained and tested using five years of historical Apple Inc. (AAPL) stock price data but you can use any stock tiker on the dashboard. 
Data analysis was performed in Python, with missing data handled through z-score of logreturn standardization. 
The z-score plot aids in identifying outliers and visually inspecting 
periods of high liquidity. This project serves as a valuable 
tool for investors, aiding them in making informed decisions and effectively managing 
investment risks.
To run:Download all files and type "streamlit run ARIMA.py" in your command prompt 
Link to the project: https://tableautest-mhf3a9ap8vqlrjci37nurw.streamlit.app/
Make sure to select 'Start Data' atlest 5 years older or more to fit enough data in the model
