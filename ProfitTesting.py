import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np

rollingWindow = 60    # days used to compute each z-score and hedge ratio
entryZScore = 2       # how far the spread must diverge before opening a trade
exitZScore = 0.5      # how close to zero the z-score must return before closing
maxHoldDays = 63      # hard cap on how long a position stays open
backtestYears = 1     # how far back the backtest starts (match yearsForwardGap in the screener)

url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
table = pd.read_csv(url)

fixed_tickers = []

for i in range(len(table)): # reformats tickers so that they can be used by yfinance
    ticker = table["Symbol"].iloc[i].replace(".", "-")
    sector = table["GICS Sector"].iloc[i]
    fixed_tickers.append(ticker)

tickers = fixed_tickers

end_date = datetime.today().strftime("%Y-%m-%d") 
start_date = (pd.Timestamp.today() - pd.DateOffset(years=backtestYears)).strftime("%Y-%m-%d") #makes start date backtestYears ago from now


def profitTesting(ticker1, ticker2, start_date, end_date): # calculates the z-score between two tickers for a certain time period
    prices = yf.download([ticker1, ticker2], start=start_date, end=end_date, auto_adjust=True)["Close"]

    pair_prices = prices[[ticker1, ticker2]].dropna()

    if (pair_prices <= 0).any().any():
        return None, None
    
    allZScores = []
    allHedgeRatios = []
    allZScores = [np.nan] * (rollingWindow - 1) #fills the first rollingWindow-1 values of the list with placeholders so that positive/negativeZScores and allZscores match in day count
    allHedgeRatios = [np.nan] * (rollingWindow - 1) #fills the first rollingWindow-1 values of the list with placeholders so that positive/negativeZScores and allHedgeRatios match in day count

    for i in range(rollingWindow - 1, len(pair_prices)): # starts at rollingWindow-1 because that's when the first full window is available

        logPrices1 = np.log(pair_prices[ticker1].iloc[i - (rollingWindow - 1):i+1]) # only takes the log prices of the ticker in the last rollingWindow days
        logPrices2 = np.log(pair_prices[ticker2].iloc[i - (rollingWindow - 1):i+1]) # only takes the log prices of the ticker in the last rollingWindow days

        mean1 = logPrices1.mean() # now we use the new regression formula with the intercept not starting at 0
        mean2 = logPrices2.mean()

        centered1 = logPrices1 - mean1
        centered2 = logPrices2 - mean2

        hedgeRatio = (centered1 * centered2).sum() / (centered2 ** 2).sum()

        alpha = mean1 - hedgeRatio * mean2

        spreads = (logPrices1 - alpha - hedgeRatio * logPrices2).tolist()

        avgSpread = np.mean(spreads)
        standardDeviation = np.std(spreads) 

        zScore = float((spreads[-1] - avgSpread) / standardDeviation) # now it calculates each z score with data from the last 60 days not from the whole data set

        allZScores.append(zScore)
        allHedgeRatios.append(hedgeRatio) # now also tracks the daily hedge ratio as it's not static anymore

    positiveZScores = {}
    negativeZScores = {}

    for i in range(len(allZScores)): # finds days with a z score above 2 and sorts them into positive and negative 

        if abs(allZScores[i]) >= entryZScore and allZScores[i] >= 0:
            positiveZScores[i] = allZScores[i] # sell stock 1 and buy stock 2
        elif abs(allZScores[i]) >= entryZScore and allZScores[i] <= 0:
            negativeZScores[i] = allZScores[i] # buy stock 1 and sell stock 2
    
    totalProfitPos = []

    freeUntil = -1 #allows us to skip days where our position is already opened to avoid bying into the same postion on consecutive days

    for key in positiveZScores.keys(): # goes through all the days with positive big z-scores
        if key + maxHoldDays >= len(pair_prices) or key < freeUntil:
            continue
        
        daysPassed = 0

        
        sellingPrice1 = pair_prices[ticker1].iloc[key] # shorts stock 1
        buyingPrice2 = pair_prices[ticker2].iloc[key] # goes long on stock 2

        while daysPassed != maxHoldDays and allZScores[key+daysPassed] >= exitZScore: # doesn't exit positions until maxHoldDays have passed or the z-score has reverted
            daysPassed += 1
            continue

        freeUntil = key + daysPassed

        buyingPrice1 = pair_prices[ticker1].iloc[key+daysPassed] # buys back ticker 1 when either of those two conditions is met
        profit1 = (sellingPrice1 - buyingPrice1) / sellingPrice1

        sellingPrice2 = pair_prices[ticker2].iloc[key+daysPassed] # sells ticker 2 when either of those two conditions is met
        profit2 = ((sellingPrice2 - buyingPrice2) / buyingPrice2) * allHedgeRatios[key]

        totalProfitPos.append(((profit1 + profit2) / (1 + abs(allHedgeRatios[key]))) * 100)

    meanProfitPos = np.mean(totalProfitPos) if totalProfitPos else 0.0
    
    totalProfitNeg = []

    freeUntil = -1 #allows us to skip days where our position is already opened to avoid bying into the same postion on consecutive days

    for key in negativeZScores.keys(): # goes through all the days with negative big z-scores
        if key + maxHoldDays >= len(pair_prices) or key < freeUntil:
            continue

        daysPassed = 0

        buyingPrice1 = pair_prices[ticker1].iloc[key] # goes long on stock 1
        sellingPrice2 = pair_prices[ticker2].iloc[key] # shorts stock 2

        while daysPassed != maxHoldDays and allZScores[key+daysPassed] <= -exitZScore: # doesn't exit positions until maxHoldDays have passed or the z-score has reverted # doesn't exit positions until maxHoldDays have passed or the z-score has reverted
        
            daysPassed += 1
            continue
        
        freeUntil = key + daysPassed

        sellingPrice1 = pair_prices[ticker1].iloc[key+daysPassed] # sells ticker 1 when either of those two conditions is met
        profit1 = (sellingPrice1 - buyingPrice1) / buyingPrice1

        buyingPrice2 = pair_prices[ticker2].iloc[key+daysPassed] # buys back ticker 2 when either of those two conditions is met
        profit2 = ((sellingPrice2 - buyingPrice2) / sellingPrice2) * allHedgeRatios[key]

        totalProfitNeg.append(((profit1 + profit2) / (1 + abs(allHedgeRatios[key]))) * 100)

    meanProfitNeg = np.mean(totalProfitNeg) if totalProfitNeg else 0.0

    return float(meanProfitPos), float(meanProfitNeg), len(totalProfitPos), len(totalProfitNeg) #added length to see how many trades were actually completed so that we can see how realiable the strategy actually is





    



