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


def profitTesting(ticker1, ticker2, start_date, end_date): #calculates the z-score between two tickers for a certain time period
    prices = yf.download([ticker1, ticker2], start=start_date, end=end_date, auto_adjust=True)["Close"]

    pair_prices = prices[[ticker1, ticker2]].dropna()

    if (pair_prices <= 0).any().any():
        return None, None

    logPrices1 = np.log(pair_prices[ticker1])
    logPrices2 = np.log(pair_prices[ticker2])

    hedgeRatio = (logPrices1 * logPrices2).sum() / (logPrices2 ** 2).sum()

    spreads = (logPrices1 - hedgeRatio * logPrices2).tolist()

    avgSpread = np.mean(spreads)
    standardDeviation = np.std(spreads) 

    positiveZScores = {}
    negativeZScores = {}
    allZScores = []

    for i in range(len(spreads)): #finds days with a z score above 2 and sorts them into positive and negative 
        zScore = float((spreads[i] - avgSpread) / standardDeviation)

        allZScores.append(zScore)

        if abs(zScore) >= 2 and zScore >= 0:
            positiveZScores[i] = zScore #sell stock 1 and buy stock 2
        elif abs(zScore) >= 2 and zScore <= 0:
            negativeZScores[i] = zScore #buy stock 1 and sell stock 2
    
    totalProfitPos = []

    for key in positiveZScores.keys(): # goes through all the days with positive big z-scores
        if key + 63 >= len(pair_prices):
            continue
        
        daysPassed = 0

        
        sellingPrice1 = pair_prices[ticker1].iloc[key] # shorts stock 1
        buyingPrice2 = pair_prices[ticker2].iloc[key] # goes long on stock 2

        while daysPassed != 63 and allZScores[key+daysPassed] >= 0.05: # doesn't exit postions until 63 trading days(3 months) have passed or the z-score has reverted
            daysPassed += 1
            continue

        buyingPrice1 = pair_prices[ticker1].iloc[key+daysPassed] # buys back ticker 1 when either of those two conditions is met
        profit1 = (sellingPrice1 - buyingPrice1) / sellingPrice1

        sellingPrice2 = pair_prices[ticker2].iloc[key+daysPassed] # sells ticker 2 when either of those two conditions is met
        profit2 = ((sellingPrice2 - buyingPrice2) / buyingPrice2) * hedgeRatio

        totalProfitPos.append(((profit1 + profit2) / (1 + hedgeRatio)) * 100)

    meanProfitPos = np.mean(totalProfitPos)
    
    totalProfitNeg = []

    for key in negativeZScores.keys(): # goes through all the days with negative big z-scores
        if key + 63 >= len(pair_prices):
            continue

        daysPassed = 0

        buyingPrice1 = pair_prices[ticker1].iloc[key] # goes long on stock 1
        sellingPrice2 = pair_prices[ticker2].iloc[key] # shorts stock 2

        while daysPassed != 63 and allZScores[key+daysPassed] <= -0.05: # doesn't exit postions until 63 trading days(3 months) have passed or the z-score has reverted
            daysPassed += 1
            continue

        sellingPrice1 = pair_prices[ticker1].iloc[key+daysPassed] # sells ticker 1 when either of those two conditions is met
        profit1 = (sellingPrice1 - buyingPrice1) / buyingPrice1

        buyingPrice2 = pair_prices[ticker2].iloc[key+daysPassed] # buys back ticker 2 when either of those two conditions is met
        profit2 = ((sellingPrice2 - buyingPrice2) / sellingPrice2) * hedgeRatio

        totalProfitNeg.append(((profit1 + profit2) / (1 + hedgeRatio)) * 100)

    meanProfitNeg = np.mean(totalProfitNeg)

    return float(meanProfitPos), float(meanProfitNeg)




print(profitTesting('GILD','PFE',start_date, end_date))
    
    



