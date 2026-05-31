import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

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
    returns1 = []
    returns2 = []

    for i in range(1, len(prices)): #For each day, calculate the percentage price change for both stocks and store them in two lists.
        returns1.append((prices[ticker1].iloc[i] - prices[ticker1].iloc[i-1]) / prices[ticker1].iloc[i-1])
        returns2.append((prices[ticker2].iloc[i] - prices[ticker2].iloc[i-1]) / prices[ticker2].iloc[i-1])

    numeratorSum = 0
    denominatorSum = 0

    for z in range(len(returns1)): #calculates hedgeratio between the tickers
        numeratorSum = (returns1[z] * returns2[z]) + numeratorSum
        denominatorSum = (returns2[z] ** 2) + denominatorSum

    hedgeRatio = numeratorSum / denominatorSum

    spreads =[]

    for z in range(0, len(returns1)): #makes a list of all the daily spreads
        spread = returns1[z] - hedgeRatio * returns2[z]
        spreads.append(spread)

    total = 0

    for spread in spreads: 
        total = total + spread
    
    avgSpread = total / len(spreads)

    varianceNumerator = 0

    for spread in spreads:
        varianceNumerator = varianceNumerator +((spread - avgSpread) **2)
    
    variance = varianceNumerator / len(spreads)

    standardDeviation = variance ** 0.5

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

        sellingPrice = prices[ticker1].iloc[key] # shorts stock 1 for a week
        buyingPrice = prices[ticker1].iloc[key+time]
        profit1 = (sellingPrice - buyingPrice) / sellingPrice

        buyingPrice = prices[ticker2].iloc[key] # goes long on stock 2 for a week
        sellingPrice = prices[ticker2].iloc[key+time]
        profit2 = ((sellingPrice - buyingPrice) / buyingPrice) * hedgeRatio

        totalProfitPos.append(((profit1 + profit2) / (1 + hedgeRatio)) * 100)

    total = 0

    for value in totalProfitPos: # calculates mean profit of buying on each positive z score
        total = total + value
    meanProfitPos = total / len(totalProfitPos)
    
    totalProfitNeg = []

    for key in negativeZScores.keys(): # goes through all the days with negative big z scores
        if key + time >= len(prices):
            continue

        buyingPrice = prices[ticker1].iloc[key] # goes long on stock 1 for a week
        sellingPrice = prices[ticker1].iloc[key+time]
        profit1 = (sellingPrice - buyingPrice) / buyingPrice

        sellingPrice = prices[ticker2].iloc[key] # shorts stock 2 for a week
        buyingPrice = prices[ticker2].iloc[key+time]
        profit2 = ((sellingPrice - buyingPrice) / sellingPrice) * hedgeRatio

        totalProfitNeg.append(((profit1 + profit2) / (1 + hedgeRatio)) * 100)

    total = 0

    for value in totalProfitNeg: # calculates mean profit of buying on each negative z score
        total = total + value
    meanProfitNeg = total / len(totalProfitNeg)

    return meanProfitPos, meanProfitNeg




print(profitTesting('JPM','BAC',start_date, end_date, 7))
    
    



