import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller

folder = os.path.dirname(os.path.abspath(__file__)) # the folder this script lives in so the cached files are found no matter where you run from

yearsBack = 2         # how far back the cointegration window starts
yearsForwardGap = 1   # how many years ago the window ends (i.e. leaves room for the backtest)
pValueCutoff = 0.05   # how strict the cointegration test is
minObsFraction = 0.8  # required fraction of expected trading days
topN = 10             # how many pairs to keep

allPrices = pd.read_pickle(os.path.join(folder, "prices.pkl")) # the 10 year price cache built by DownloadPrices.py
sectors = pd.read_pickle(os.path.join(folder, "sectors.pkl")) # the sector groupings built by DownloadPrices.py


def screenPairs(start_date, end_date): # runs the cointegration scan over one window and returns the best pairs
    minObs = int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365 * 252 * minObsFraction) # scales the overlapping data requirment to the number of days we sample

    prices = allPrices.loc[start_date:end_date] # slices the cache down to the screening window instead of downloading

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

    return [pair for pair, value in top10] # just the ticker pairs, ready to hand to the backtest


if __name__ == "__main__": # only runs when this file is executed directly, not when it's imported
    end_date = (pd.Timestamp.today() - pd.DateOffset(years=yearsForwardGap)).strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=yearsBack)).strftime("%Y-%m-%d")

    for pair in screenPairs(start_date, end_date):
        print(pair)












                













            





