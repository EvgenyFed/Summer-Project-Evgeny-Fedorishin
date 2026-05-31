import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
from statsmodels.tsa.stattools import adfuller

url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
table = pd.read_csv(url)

fixed_tickers = []
sectors = {}

for i in range(len(table)): #reformats tickers so that they can be used by yfinance
    ticker = table["Symbol"].iloc[i].replace(".", "-")
    sector = table["GICS Sector"].iloc[i]
    fixed_tickers.append(ticker)

    if sector not in sectors:
        sectors[sector] = []

    sectors[sector].append(ticker) #makes a list of all the sectors

tickers = fixed_tickers


end_date = datetime.today().strftime("%Y-%m-%d") 
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d") #makes start data exactly a year ago from now

prices = yf.download(tickers, start = start_date, end = end_date, auto_adjust= True)["Close"]

beta = {}



for sector in sectors: #used to cycle through sectors
    for j in range(len(sectors[sector])): #used to cycle through first tickers
        for k in range(j+1, len(sectors[sector])): #used to cycle through second tickers

            ticker1 = sectors[sector][j]
            ticker2 = sectors[sector][k]

            logPrices1 = np.log(prices[ticker1].dropna())
            logPrices2 = np.log(prices[ticker2].dropna())

            if logPrices1.isin([np.inf, -np.inf]).any() or logPrices2.isin([np.inf, -np.inf]).any():
                continue

            if len(logPrices1) != len(logPrices2):
                continue

            if ticker1 not in prices.columns or ticker2 not in prices.columns:
                continue

            hedgeRatio = (logPrices1 * logPrices2).sum() / (logPrices2 ** 2).sum()
            spread = logPrices1 - hedgeRatio * logPrices2

            adfStat = adfuller(spread)[0]
            beta[ticker1, ticker2] = adfStat

def getAbsBeta(item): 
    return abs(item[1])

sortedBeta = sorted(beta.items(), key=getAbsBeta) 

top10 = sortedBeta[-10:] #list of 10 most correlated pairs

for pair, value in top10:
    print(pair, value)












                













            





