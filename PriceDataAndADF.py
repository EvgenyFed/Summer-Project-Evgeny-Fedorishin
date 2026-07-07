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
pvals ={}



for sector in sectors: #used to cycle through sectors
    for j in range(len(sectors[sector])): #used to cycle through first tickers
        for k in range(j+1, len(sectors[sector])): #used to cycle through second tickers

            ticker1 = sectors[sector][j]
            ticker2 = sectors[sector][k]

            if ticker1 not in prices.columns or ticker2 not in prices.columns:
                continue

            pair_prices = prices[[ticker1, ticker2]].dropna() #if price isn't available for one ticker on a certain date drops the other tickers price on that date as well in order to not scew the data
            
            if len(pair_prices) < 200: #created a minimum overlapping data requirment
                continue

            if (pair_prices <= 0).any().any():
                continue

            logPrices1 = np.log(pair_prices[ticker1])
            logPrices2 = np.log(pair_prices[ticker2])

            hedgeRatio = (logPrices1 * logPrices2).sum() / (logPrices2 ** 2).sum()
            spread = logPrices1 - hedgeRatio * logPrices2

            adfStat, pValue, _, _, _, _ = adfuller(spread) #extract the pValue as well now

            if pValue <= 0.05: #added the pValue check for signficance testing
                beta[ticker1, ticker2] = adfStat
                pvals[ticker1, ticker2] = pValue

def getPValue(item): 
    return pvals[item[0]]

sortedBeta = sorted(beta.items(), key=getPValue)

top10 = sortedBeta[:10]



for pair, value in top10:
    print(pair, value)












                













            





