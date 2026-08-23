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
recheckMonths = 1          # months between follow-up cointegration tests
recheckLookbackYears = 1   # how much trailing history each recheck tests on
pValueCutoff = 0.05        # recheck fails if the ADF p-value exceeds this

end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (pd.Timestamp.today() - pd.DateOffset(years=backtestYears)).strftime("%Y-%m-%d") #makes start date backtestYears ago from now


def profitTesting(ticker1, ticker2, pair_prices, start_date): # calculates the z-score between two tickers over prices passed in from outside
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
    
    allSignals = {}
    allSignals.update(positiveZScores)
    allSignals.update(negativeZScores) # merges both directions so one pass can walk them in date order

    totalProfitPos = []
    totalProfitNeg = []

    freeUntil = -1 #allows us to skip days where our position is already opened to avoid bying into the same postion on consecutive days

    dailyPnL = pd.Series(0.0, index=pair_prices.index) # how much the pair gained or lost each day, indexed by date so pairs can be lined up later, stays 0 when no trade is open

    for key in sorted(allSignals.keys()): # goes through every signal day in date order regardless of direction
        if key + maxHoldDays >= len(pair_prices) or key < freeUntil:
            continue

        daysPassed = 0

        if allSignals[key] >= 0: # positive z-score so short stock 1 and long stock 2

            sellingPrice1 = pair_prices[ticker1].iloc[key] # shorts stock 1
            buyingPrice2 = pair_prices[ticker2].iloc[key] # goes long on stock 2

            while daysPassed != maxHoldDays and key + daysPassed < deathDay and allZScores[key+daysPassed] >= exitZScore: # doesn't exit postions until 63 trading days(3 months) have passed or the z-score has reverted
                daysPassed += 1
                continue

            freeUntil = key + daysPassed

            buyingPrice1 = pair_prices[ticker1].iloc[key+daysPassed] # buys back ticker 1 when either of those two conditions is met
            profit1 = (sellingPrice1 - buyingPrice1) / sellingPrice1

            sellingPrice2 = pair_prices[ticker2].iloc[key+daysPassed] # sells ticker 2 when either of those two conditions is met
            profit2 = ((sellingPrice2 - buyingPrice2) / buyingPrice2) * allHedgeRatios[key]

            totalProfitPos.append(((profit1 + profit2) / (1 + abs(allHedgeRatios[key]))) * 100)

            cumulative = 0.0

            for d in range(key + 1, key + daysPassed + 1): # walks through each day the trade was open to record its daily change
                runningProfit1 = (sellingPrice1 - pair_prices[ticker1].iloc[d]) / sellingPrice1
                runningProfit2 = ((pair_prices[ticker2].iloc[d] - buyingPrice2) / buyingPrice2) * allHedgeRatios[key]

                newCumulative = ((runningProfit1 + runningProfit2) / (1 + abs(allHedgeRatios[key]))) * 100

                dailyPnL.iloc[d] = newCumulative - cumulative # only the change since yesterday, not the running total
                cumulative = newCumulative

        else: # negative z-score so long stock 1 and short stock 2

            buyingPrice1 = pair_prices[ticker1].iloc[key] # goes long on stock 1
            sellingPrice2 = pair_prices[ticker2].iloc[key] # shorts stock 2

            while daysPassed != maxHoldDays and key + daysPassed < deathDay and allZScores[key+daysPassed] <= -exitZScore: # doesn't exit postions until 63 trading days(3 months) have passed or the z-score has reverted
                daysPassed += 1
                continue

            freeUntil = key + daysPassed

            sellingPrice1 = pair_prices[ticker1].iloc[key+daysPassed] # sells ticker 1 when either of those two conditions is met
            profit1 = (sellingPrice1 - buyingPrice1) / buyingPrice1

            buyingPrice2 = pair_prices[ticker2].iloc[key+daysPassed] # buys back ticker 2 when either of those two conditions is met
            profit2 = ((sellingPrice2 - buyingPrice2) / sellingPrice2) * allHedgeRatios[key]

            totalProfitNeg.append(((profit1 + profit2) / (1 + abs(allHedgeRatios[key]))) * 100)

            cumulative = 0.0

            for d in range(key + 1, key + daysPassed + 1): # walks through each day the trade was open to record its daily change
                runningProfit1 = (pair_prices[ticker1].iloc[d] - buyingPrice1) / buyingPrice1
                runningProfit2 = ((sellingPrice2 - pair_prices[ticker2].iloc[d]) / sellingPrice2) * allHedgeRatios[key]

                newCumulative = ((runningProfit1 + runningProfit2) / (1 + abs(allHedgeRatios[key]))) * 100

                dailyPnL.iloc[d] = newCumulative - cumulative # only the change since yesterday, not the running total
                cumulative = newCumulative

    meanProfitPos = np.mean(totalProfitPos) if totalProfitPos else 0.0
    meanProfitNeg = np.mean(totalProfitNeg) if totalProfitNeg else 0.0

    return float(meanProfitPos), float(meanProfitNeg), len(totalProfitPos), len(totalProfitNeg), deathDay, len(pair_prices), dailyPnL, totalProfitPos + totalProfitNeg #added length to see how many trades were actually completed so that we can see how realiable the strategy actually is, plus deathDay to check the recheck logic, dailyPnL for the equity curve and every individual trade profit for the win rate

if __name__ == "__main__": # only runs when this file is executed directly, not when it's imported
    downloadStart = (pd.to_datetime(start_date) - pd.DateOffset(years=recheckLookbackYears)).strftime("%Y-%m-%d") # pulls extra history so the first recheck has a full lookback behind it

    prices = yf.download(['MSCI','PYPL'], start=downloadStart, end=end_date, auto_adjust=True)["Close"]

    pair_prices = prices[['MSCI','PYPL']].dropna()

    print(profitTesting('MSCI','PYPL', pair_prices, start_date))



    



