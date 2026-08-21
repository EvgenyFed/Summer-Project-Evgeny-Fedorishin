import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller

rollingWindow = 60    # days used to compute each z-score and hedge ratio
entryZScore = 2       # how far the spread must diverge before opening a trade
exitZScore = 0.5      # how close to zero the z-score must return before closing
maxHoldDays = 63      # hard cap on how long a position stays open
backtestYears = 1     # how far back the backtest starts (match yearsForwardGap in the screener)
recheckMonths = 2          # months between follow-up cointegration tests
recheckLookbackYears = 1   # how much trailing history each recheck tests on
pValueCutoff = 0.05        # recheck fails if the ADF p-value exceeds this

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
    downloadStart = (pd.to_datetime(start_date) - pd.DateOffset(years=recheckLookbackYears)).strftime("%Y-%m-%d") # pulls extra history so the first recheck has a full lookback behind it

    prices = yf.download([ticker1, ticker2], start=downloadStart, end=end_date, auto_adjust=True)["Close"]

    pair_prices = prices[[ticker1, ticker2]].dropna()

    backtestStart = pair_prices.index.searchsorted(pd.to_datetime(start_date)) # converts start_date into a row number so we know where the extra downloaded year ends and the tradeable period begins

    if (pair_prices <= 0).any().any():
        return None, None, 0, 0, 0, 0
    
    allZScores = []
    allHedgeRatios = []
    allZScores = [np.nan] * (rollingWindow - 1) #fills the first rollingWindow-1 values of the list with placeholders so that positive/negativeZScores and allZscores match in day count
    allHedgeRatios = [np.nan] * (rollingWindow - 1) #fills the first rollingWindow-1 values of the list with placeholders so that positive/negativeZScores and allHedgeRatios match in day count

    for i in range(rollingWindow - 1, len(pair_prices)): # starts at rollingWindow-1 because that's when the first full window is available

        logPrices1 = np.log(pair_prices[ticker1].iloc[i - (rollingWindow - 1):i+1]) # only takes the log prices of the ticker in the last rollingWindow days
        logPrices2 = np.log(pair_prices[ticker2].iloc[i - (rollingWindow - 1):i+1]) # only takes the log prices of the ticker in the last rollingWindow days

        mean1 = logPrices1.mean() # the new regression formula with the intercept not starting at 0
        mean2 = logPrices2.mean()

        centered1 = logPrices1 - mean1
        centered2 = logPrices2 - mean2

        hedgeRatio = (centered1 * centered2).sum() / (centered2 ** 2).sum()

        alpha = mean1 - hedgeRatio * mean2

        spreads = (logPrices1 - alpha - hedgeRatio * logPrices2).tolist()

        avgSpread = np.mean(spreads)
        standardDeviation = np.std(spreads) 

        zScore = float((spreads[-1] - avgSpread) / standardDeviation) # calculates each z score with data from the last 60 days not from the whole data set

        allZScores.append(zScore)
        allHedgeRatios.append(hedgeRatio) # tracks the daily hedge ratio as it's not static anymore
    
    lookbackDays = int(recheckLookbackYears * 252)
    recheckSpacing = int(recheckMonths * 21)

    deathDay = len(pair_prices) # the day the pair stops being tradeable, defaults to never

    for d in range(backtestStart, len(pair_prices), recheckSpacing): # reruns the cointegration test every recheckMonths
        windowStart = max(0, d - lookbackDays + 1)

        recheckLog1 = np.log(pair_prices[ticker1].iloc[windowStart:d+1])
        recheckLog2 = np.log(pair_prices[ticker2].iloc[windowStart:d+1])

        rMean1 = recheckLog1.mean()
        rMean2 = recheckLog2.mean()

        rHedgeRatio = ((recheckLog1 - rMean1) * (recheckLog2 - rMean2)).sum() / ((recheckLog2 - rMean2) ** 2).sum()
        rAlpha = rMean1 - rHedgeRatio * rMean2

        recheckSpread = recheckLog1 - rAlpha - rHedgeRatio * recheckLog2

        _, recheckPValue, _, _, _, _ = adfuller(recheckSpread)

        if recheckPValue > pValueCutoff: # pair is no longer cointegrated so it dies here and stays dead
            deathDay = d
            break

    positiveZScores = {}
    negativeZScores = {}

    for i in range(backtestStart, deathDay): # now only signals high z-score inside the real profit testing period and before the pair dies 

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

        while daysPassed != maxHoldDays and key + daysPassed < deathDay and allZScores[key+daysPassed] >= exitZScore: # doesn't exit positions until maxHoldDays have passed or the z-score has reverted also exits if the pair fails its cointegration recheck
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

    freeUntil = -1 # allows us to skip days where our position is already opened to avoid bying into the same postion on consecutive days

    for key in negativeZScores.keys(): # goes through all the days with negative big z-scores
        if key + maxHoldDays >= len(pair_prices) or key < freeUntil:
            continue

        daysPassed = 0

        buyingPrice1 = pair_prices[ticker1].iloc[key] # goes long on stock 1
        sellingPrice2 = pair_prices[ticker2].iloc[key] # shorts stock 2

        while daysPassed != maxHoldDays and key + daysPassed < deathDay and allZScores[key+daysPassed] <= -exitZScore: # doesn't exit positions until maxHoldDays have passed or the z-score has reverted also exits if the pair fails its cointegration recheck
        
            daysPassed += 1
            continue
        
        freeUntil = key + daysPassed

        sellingPrice1 = pair_prices[ticker1].iloc[key+daysPassed] # sells ticker 1 when either of those two conditions is met
        profit1 = (sellingPrice1 - buyingPrice1) / buyingPrice1

        buyingPrice2 = pair_prices[ticker2].iloc[key+daysPassed] # buys back ticker 2 when either of those two conditions is met
        profit2 = ((sellingPrice2 - buyingPrice2) / sellingPrice2) * allHedgeRatios[key]

        totalProfitNeg.append(((profit1 + profit2) / (1 + abs(allHedgeRatios[key]))) * 100)

    meanProfitNeg = np.mean(totalProfitNeg) if totalProfitNeg else 0.0

    return float(meanProfitPos), float(meanProfitNeg), len(totalProfitPos), len(totalProfitNeg), deathDay, len(pair_prices) #added length to see how many trades were actually completed so that we can see how realiable the strategy actually is, plus deathDay to check the recheck logic

print(profitTesting('MSCI','PYPL',start_date, end_date))



    



