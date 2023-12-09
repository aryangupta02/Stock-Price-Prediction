import streamlit as st
from datetime import date
import datetime
import yfinance as yf

from prophet import Prophet, forecaster
    

from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2017-01-01"
TODAY = (date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d");


st.set_page_config(page_title='StockPi: Predict Stocks', layout='wide', menu_items=None)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("StockPi")

stocks = ["Reliance - RELIANCE.NS", "Apple - AAPL", "Google - GOOG", "Microsoft - MSFT", "Tata Consultancy Service - TCS.NS", "NIfty 50 - NSEI"]
st.table(stocks)
selected_stock = st.text_input("Enter Stock symbol from above or by yourself", value="")

period = st.number_input("Select the number of days to predict", min_value=100, max_value=365, value=100)

clicked = st.button("Predict")

if clicked: 
    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    with st.spinner(text="Loading Data..."):
        data = load_data(selected_stock)
        if(data[data.columns[0]].count()==0):
            st.error("Only Stock Symbols allowed")
        else:
            st.success("Raw Data fetched!")
    if(data[data.columns[0]].count()>0): 
        st.subheader("Raw Stock Data")
        st.write(data.tail())

        st.subheader("Time Seried Data")
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open Price"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close Price"))
            fig.layout.update(title_text = "RAW Data for "+selected_stock, xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)

        plot_raw_data()


        #Forecasting
        with st.spinner(text="Prediction in Progress..."):
            df_train = data[['Date', 'Close']]
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

            m = Prophet()

            m.fit(df_train)

            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)


            
        st.success("Prediction Complete. Checkout the result!")
        st.subheader("Predicted Stock Data")
        st.write(forecast.tail(5))

        st.write("Forecast data")
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        fig2 = m.plot_components(forecast)
        st.write(fig2)