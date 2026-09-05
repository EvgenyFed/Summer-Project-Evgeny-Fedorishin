import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
import Config

folder = os.path.dirname(os.path.abspath(__file__)) # the folder this script lives in so the cached files are found no matter where you run from

yearsBack = 2         # how far back the cointegration window starts
yearsForwardGap = 1   # how many years ago the window ends (i.e. leaves room for the backtest)
pValueCutoff = 0.05   # how strict the cointegration test is
minObsFraction = 0.8  # required fraction of expected trading days
topN = 10             # how many pairs to keep

allPrices = pd.read_pickle(os.path.join(folder, "prices.pkl")) # the 10 year price cache built by DownloadPrices.py
sectors = pd.read_pickle(os.path.join(folder, "sectors.pkl")) # the sector groupings built by DownloadPrices.py


def fitPair(logPrices1, logPrices2): # one regression: returns the intercept, hedge ratio and residual spread
    mean1 = logPrices1.mean()
    mean2 = logPrices2.mean()

    centered1 = logPrices1 - mean1
    centered2 = logPrices2 - mean2

    denominator = (centered2 ** 2).sum()

    if denominator == 0:
        return None

    hedgeRatio = (centered1 * centered2).sum() / denominator

    alpha = mean1 - hedgeRatio * mean2

    spread = logPrices1 - alpha - hedgeRatio * logPrices2

    return alpha, hedgeRatio, spread


def halfLife(spread): # how many days a deviation takes to decay by half
    values = np.asarray(spread, dtype=float)

    if len(values) < 30:
        return np.nan

    lagged = values[:-1]
    change = np.diff(values)

    X = np.column_stack([np.ones(len(lagged)), lagged])

    coefficients, _, _, _ = np.linalg.lstsq(X, change, rcond=None)

    gamma = coefficients[1]
    phi = 1.0 + gamma

    if 0 < phi < 1: # anything else means the spread is not mean reverting at all
        return float(np.log(2.0) / -np.log(phi))

    return np.nan


def screenPairs(start_date, end_date): # runs the cointegration scan over one window and returns a frozen record per surviving pair
    minObs = int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365 * 252 * minObsFraction) # scales the overlapping data requirment to the number of days we sample

    prices = allPrices.loc[start_date:end_date] # slices the cache down to the screening window instead of downloading

    records = []

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

                fit = fitPair(logPrices1, logPrices2)

                if fit is None:
                    continue

                alpha, hedgeRatio, spread = fit

                if hedgeRatio <= 0: # a negative hedge ratio holds both legs the same way round, which is a directional bet not a pairs trade
                    continue

                if spread.std() == 0:
                    continue

                adfStat, pValue, _, _, _, _ = adfuller(spread) #extracts the pValue as well now

                if pValue > pValueCutoff: #added the pValue check for signficance testing
                    continue

                halfPoint = len(pair_prices) // 2 # the two halves used by the stability filter

                firstFit = fitPair(logPrices1.iloc[:halfPoint], logPrices2.iloc[:halfPoint])
                secondFit = fitPair(logPrices1.iloc[halfPoint:], logPrices2.iloc[halfPoint:])

                stable = False

                if firstFit is not None and secondFit is not None:
                    if firstFit[1] > 0 and secondFit[1] > 0 and firstFit[2].std() > 0 and secondFit[2].std() > 0:
                        _, firstP, _, _, _, _ = adfuller(firstFit[2])
                        _, secondP, _, _, _, _ = adfuller(secondFit[2])
                        stable = firstP <= Config.halfPValueCutoff and secondP <= Config.halfPValueCutoff

                if Config.useStability and not stable: # only bites when the flag is on, so the baseline is unaffected
                    continue

                pairHalfLife = halfLife(spread)

                if Config.useHalfLife:
                    if np.isnan(pairHalfLife):
                        continue
                    if pairHalfLife < Config.halfLifeMin or pairHalfLife > Config.halfLifeMax:
                        continue

                records.append({
                    "ticker1": ticker1,
                    "ticker2": ticker2,
                    "alpha": float(alpha),
                    "beta": float(hedgeRatio),
                    "spreadMean": float(spread.mean()),
                    "spreadStd": float(spread.std()),
                    "spreadChangeStd": float(spread.diff().dropna().std()), # the formation volatility the vol filter compares against
                    "halfLife": pairHalfLife,
                    "stable": stable,
                    "adfStat": float(adfStat),
                    "pValue": float(pValue),
                })

    def getPValue(record):
        return record["pValue"]

    records.sort(key=getPValue)

    return records[:topN] # frozen formation records, ready to hand to the backtest


if __name__ == "__main__": # only runs when this file is executed directly, not when it's imported
    end_date = (pd.Timestamp.today() - pd.DateOffset(years=yearsForwardGap)).strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=yearsBack)).strftime("%Y-%m-%d")

    for record in screenPairs(start_date, end_date):
        print(record["ticker1"], record["ticker2"],
              "beta", round(record["beta"], 3),
              "halfLife", round(record["halfLife"], 1) if not np.isnan(record["halfLife"]) else "na",
              "stable", record["stable"])












                













            





