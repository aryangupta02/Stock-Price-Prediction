
from datetime import date

import yfinance as yf

from fastapi import FastAPI
from pydantic import BaseModel

import requests


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API Home"}

class TickerPrefix(BaseModel):
    ticker: str

@app.post("/getSuggestiveTicker")
async def getTickerSuggestion(tickerPrefix: TickerPrefix):
    response = requests.get("https://finance.yahoo.com/_finance_doubledown/api/resource/searchassist;searchTerm=Reliance?device=console&returnMeta=true")
    print(response)


class PredictionInput(BaseModel):
    tickerSelected: str
    days: int

@app.post("/getPredictionClose")
async def getPrediction(predictionInput: PredictionInput):
    START = "2017-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    def load_data():
        data = yf.download(predictionInput.tickerSelected, START, TODAY)
        data.reset_index(inplace=True)
        return data
    data = load_data()

    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()

    m.fit(df_train)

    future = m.make_future_dataframe(periods=predictionInput.days)
    forecast = m.predict(future)

    return forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']].tail(predictionInput.days)

@app.post("/getPredictionOpen")
async def getPrediction(predictionInput: PredictionInput):
    START = "2017-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")
    def load_data():
        data = yf.download(predictionInput.tickerSelected, START, TODAY)
        data.reset_index(inplace=True)
        return data
    data = load_data()

    df_train = data[['Date', 'Open']]
    df_train = df_train.rename(columns={"Date": "ds", "Open": "y"})

    m = Prophet()

    m.fit(df_train)

    future = m.make_future_dataframe(periods=predictionInput.days)
    forecast = m.predict(future)

    return forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']].tail(predictionInput.days)

