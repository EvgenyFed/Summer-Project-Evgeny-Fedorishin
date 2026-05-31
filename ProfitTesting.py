import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np

url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
table = pd.read_csv(url)

fixed_tickers = []

for i in range(len(table)): #reformats tickers so that they can be used by yfinance
    ticker = table["Symbol"].iloc[i].replace(".", "-")
    sector = table["GICS Sector"].iloc[i]
    fixed_tickers.append(ticker)

tickers = fixed_tickers


end_date = datetime.today().strftime("%Y-%m-%d") 
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d") #makes start data exactly a year ago from now


def profitTesting(ticker1, ticker2, start_date, end_date, time): #calculates the z-score given two tickers and a time period between buying and selling
    prices = yf.download([ticker1, ticker2], start=start_date, end=end_date, auto_adjust=True)["Close"]

    logPrices1 = np.log(prices[ticker1].dropna())
    logPrices2 = np.log(prices[ticker2].dropna())

    hedgeRatio = (logPrices1 * logPrices2).sum() / (logPrices2 ** 2).sum()

    spreads = (logPrices1 - hedgeRatio * logPrices2).tolist()

    avgSpread = np.mean(spreads)
    standardDeviation = np.std(spreads) 

    positiveZScores = {}
    negativeZScores = {}

    for i in range(len(spreads)): #finds days with a z score above 2 and sorts them into positive and negative 
        zScore = float((spreads[i] - avgSpread) / standardDeviation)
        if abs(zScore) >= 2 and zScore >= 0:
            positiveZScores[i] = zScore #sell stock 1 and buy stock 2
        elif abs(zScore) >= 2 and zScore <= 0:
            negativeZScores[i] = zScore #buy stock 1 and sell stock 2
    
    totalProfitPos = []

    for key in positiveZScores.keys(): # goes through all the days with positive big z scores
        if key + time >= len(prices):
            continue

        sellingPrice = prices[ticker1].iloc[key] # shorts stock 1
        buyingPrice = prices[ticker1].iloc[key+time]
        profit1 = (sellingPrice - buyingPrice) / sellingPrice

        buyingPrice = prices[ticker2].iloc[key] # goes long on stock 2
        sellingPrice = prices[ticker2].iloc[key+time]
        profit2 = ((sellingPrice - buyingPrice) / buyingPrice) * hedgeRatio

        totalProfitPos.append(((profit1 + profit2) / (1 + hedgeRatio)) * 100)

    meanProfitPos = np.mean(totalProfitPos)
    
    totalProfitNeg = []

    for key in negativeZScores.keys(): # goes through all the days with negative big z scores
        if key + time >= len(prices):
            continue

        buyingPrice = prices[ticker1].iloc[key] # goes long on stock 1
        sellingPrice = prices[ticker1].iloc[key+time]
        profit1 = (sellingPrice - buyingPrice) / buyingPrice

        sellingPrice = prices[ticker2].iloc[key] # shorts stock 2
        buyingPrice = prices[ticker2].iloc[key+time]
        profit2 = ((sellingPrice - buyingPrice) / sellingPrice) * hedgeRatio

        totalProfitNeg.append(((profit1 + profit2) / (1 + hedgeRatio)) * 100)

    meanProfitNeg = np.mean(totalProfitNeg)

    return float(meanProfitPos), float(meanProfitNeg)




print(profitTesting('JPM','BAC',start_date, end_date, 7))
    
    



