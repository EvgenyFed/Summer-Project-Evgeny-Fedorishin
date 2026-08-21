import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
from statsmodels.tsa.stattools import adfuller

yearsBack = 2         # how far back the cointegration window starts
yearsForwardGap = 1   # how many years ago the window ends (i.e. leaves room for the backtest)
pValueCutoff = 0.05   # how strict the cointegration test is
minObsFraction = 0.8  # required fraction of expected trading days
topN = 10             # how many pairs to keep

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


end_date = (pd.Timestamp.today() - pd.DateOffset(years=yearsForwardGap)).strftime("%Y-%m-%d") # makes the end date yearsForwardGap years ago from now
start_date = (pd.Timestamp.today() - pd.DateOffset(years=yearsBack)).strftime("%Y-%m-%d") # makes start date yearsBack years ago from now
minObs = int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365 * 252 * minObsFraction) # scales the overlapping data requirment to the number of days we sample



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
            
            if len(pair_prices) < minObs: #created a minimum overlapping data requirment
                continue

            if (pair_prices <= 0).any().any():
                continue

            logPrices1 = np.log(pair_prices[ticker1])
            logPrices2 = np.log(pair_prices[ticker2])


            mean1 = logPrices1.mean() # now we use the new regression formula with the intercept not starting at 0
            mean2 = logPrices2.mean()

            centered1 = logPrices1 - mean1
            centered2 = logPrices2 - mean2

            hedgeRatio = (centered1 * centered2).sum() / (centered2 ** 2).sum()

            alpha = mean1 - hedgeRatio * mean2

            spread = (logPrices1 - alpha - hedgeRatio * logPrices2)
            
            adfStat, pValue, _, _, _, _ = adfuller(spread) #extracts the pValue as well now

            if pValue <= pValueCutoff: #added the pValue check for signficance testing
                beta[ticker1, ticker2] = adfStat
                pvals[ticker1, ticker2] = pValue

def getPValue(item): 
    return pvals[item[0]]

sortedBeta = sorted(beta.items(), key=getPValue)

top10 = sortedBeta[:topN]

for pair, value in top10:
    print(pair, value)












                













            





