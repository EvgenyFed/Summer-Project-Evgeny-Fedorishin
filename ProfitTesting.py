import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
import Config

rollingWindow = 60    # days used to compute each z-score and hedge ratio
entryZScore = 2       # how far the spread must diverge before opening a trade
exitZScore = 0.5      # how close to zero the z-score must return before closing
maxHoldDays = 63      # hard cap on how long a position stays open
backtestYears = 1     # how far back the backtest starts (match yearsForwardGap in the screener)
recheckMonths = 2          # months between follow-up cointegration tests
recheckLookbackYears = 2   # how much trailing history each recheck tests on
pValueCutoff = 0.05        # recheck fails if the ADF p-value exceeds this
executionDelay = 1    # trading days between signal and fill (0 = same close, 1 = next close)

end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (pd.Timestamp.today() - pd.DateOffset(years=backtestYears)).strftime("%Y-%m-%d") # makes start date backtestYears ago from now


def profitTesting(ticker1, ticker2, pair_prices, start_date, record=None): # record holds the frozen formation parameters; None falls back to the old rolling behaviour
    backtestStart = pair_prices.index.searchsorted(pd.to_datetime(start_date)) # converts start_date into a row number so we know where the extra downloaded year ends and the tradeable period begins

    if (pair_prices <= 0).any().any():
        return 0.0, 0.0, 0, 0, 0, 0, pd.Series(0.0, index=pair_prices.index), [], {}

    logPrices1 = np.log(pair_prices[ticker1])
    logPrices2 = np.log(pair_prices[ticker2])

    if Config.paramMode == "fixed" and record is not None: # alpha, beta, mean and std were frozen at the end of formation and are never re-estimated here
        beta = record["beta"]

        spreadSeries = logPrices1 - record["alpha"] - beta * logPrices2

        zSeries = (spreadSeries - record["spreadMean"]) / record["spreadStd"]

        allZScores = zSeries.tolist()
        allHedgeRatios = [beta] * len(pair_prices)

        formationChangeStd = record["spreadChangeStd"]

    else: # the original 60-day rolling window, kept so the fixed-vs-rolling comparison uses the same code path
        allZScores = [np.nan] * (rollingWindow - 1) #fills the first rollingWindow-1 values of the list with placeholders so that the z-scores match the price series in day count
        allHedgeRatios = [np.nan] * (rollingWindow - 1)

        spreadValues = [np.nan] * (rollingWindow - 1)

        for i in range(rollingWindow - 1, len(pair_prices)): # starts at rollingWindow-1 because that's when the first full window is available

            windowLog1 = logPrices1.iloc[i - (rollingWindow - 1):i+1]
            windowLog2 = logPrices2.iloc[i - (rollingWindow - 1):i+1]

            mean1 = windowLog1.mean()
            mean2 = windowLog2.mean()

            centered1 = windowLog1 - mean1
            centered2 = windowLog2 - mean2

            hedgeRatio = (centered1 * centered2).sum() / (centered2 ** 2).sum()

            alpha = mean1 - hedgeRatio * mean2

            spreads = (windowLog1 - alpha - hedgeRatio * windowLog2).tolist()

            standardDeviation = np.std(spreads)

            if standardDeviation == 0:
                allZScores.append(np.nan)
                allHedgeRatios.append(hedgeRatio)
                spreadValues.append(spreads[-1])
                continue

            allZScores.append(float((spreads[-1] - np.mean(spreads)) / standardDeviation))
            allHedgeRatios.append(hedgeRatio)
            spreadValues.append(spreads[-1])

        spreadSeries = pd.Series(spreadValues, index=pair_prices.index)

        formationChangeStd = spreadSeries.diff().iloc[:backtestStart].std()

    volRatio = pd.Series(np.nan, index=pair_prices.index) # recent spread volatility measured against the formation period

    if formationChangeStd and formationChangeStd > 0:
        recentVol = spreadSeries.diff().rolling(Config.volWindow).std().shift(1) # the shift keeps today's move out of today's entry decision
        volRatio = recentVol / formationChangeStd

    volRatioValues = volRatio.tolist()
    
    lookbackDays = int(recheckLookbackYears * 252)
    recheckSpacing = int(recheckMonths * 21)

    deathDay = len(pair_prices) # the day the pair stops being tradeable, defaults to never

    recheckDays = range(backtestStart + recheckSpacing, len(pair_prices), recheckSpacing) if Config.recheckMode != "off" else []

    for d in recheckDays:
        windowStart = max(0, d - lookbackDays + 1)

        recheckLog1 = logPrices1.iloc[windowStart:d+1]
        recheckLog2 = logPrices2.iloc[windowStart:d+1]

        rMean1 = recheckLog1.mean()
        rMean2 = recheckLog2.mean()

        rHedgeRatio = ((recheckLog1 - rMean1) * (recheckLog2 - rMean2)).sum() / ((recheckLog2 - rMean2) ** 2).sum()
        rAlpha = rMean1 - rHedgeRatio * rMean2

        recheckSpread = recheckLog1 - rAlpha - rHedgeRatio * recheckLog2

        if recheckSpread.std() == 0:
            deathDay = d
            break

        _, recheckPValue, _, _, _, _ = adfuller(recheckSpread)

        if recheckPValue > pValueCutoff: # pair is no longer cointegrated so it dies here and stays dead
            deathDay = d
            break

    allSignals = {}

    for i in range(backtestStart, deathDay): # now only signals high z-score inside the real profit testing period and before the pair dies
        z = allZScores[i]

        if z is None or np.isnan(z):
            continue

        if abs(z) >= entryZScore:
            allSignals[i] = z # positive means sell stock 1 and buy stock 2, negative means the other way round

    totalProfitPos = []
    totalProfitNeg = []
    exitReasons = {}

    freeUntil = -1 #allows us to skip days where our position is already opened to avoid bying into the same postion on consecutive days

    dailyPnL = pd.Series(0.0, index=pair_prices.index) # how much the pair gained or lost each day, indexed by date so pairs can be lined up later, stays 0 when no trade is open

    lastTradableDay = len(pair_prices) - 1 - executionDelay

    for key in sorted(allSignals.keys()): # goes through every signal day in date order regardless of direction

        if key > lastTradableDay or key < freeUntil:
            continue

        if Config.useZStop and abs(allSignals[key]) >= Config.zStopLevel: # already past the stop level, so don't open a trade we'd immediately stop out of
            continue

        if Config.useVolFilter: # blocks NEW entries only, open positions keep running on the normal exit rules
            ratio = volRatioValues[key]
            if not np.isnan(ratio) and ratio > Config.volThreshold:
                continue

        direction = -1 if allSignals[key] >= 0 else 1 # -1 shorts stock 1 and longs stock 2, +1 does the reverse

        beta = allHedgeRatios[key]

        if beta is None or np.isnan(beta) or beta <= 0:
            continue

        daysPassed = 0
        exitReason = "timeLimit"

        while True: # walks forward day by day until one of the exit rules fires
            if daysPassed == maxHoldDays:
                exitReason = "timeLimit"
                break

            if Config.recheckMode == "exit" and key + daysPassed >= deathDay: # blockEntry lets open trades run to their normal exit instead of crystallising the loss here
                exitReason = "pairDeath"
                break

            if key + daysPassed > lastTradableDay:
                exitReason = "endOfData"
                break

            z = allZScores[key + daysPassed]

            if np.isnan(z):
                exitReason = "noSignal"
                break

            if Config.useZStop and daysPassed >= 1 and abs(z) >= Config.zStopLevel: # observed at today's close, filled at the next close like every other signal
                exitReason = "zStop"
                break

            if direction == -1 and z < exitZScore:
                exitReason = "converged"
                break

            if direction == 1 and z > -exitZScore:
                exitReason = "converged"
                break

            daysPassed += 1

        entryDay = key + executionDelay
        exitDay = min(key + daysPassed + executionDelay, len(pair_prices) - 1)

        if exitDay <= entryDay:
            continue

        freeUntil = exitDay

        entryPrice1 = pair_prices[ticker1].iloc[entryDay]
        entryPrice2 = pair_prices[ticker2].iloc[entryDay]

        def markedReturn(day): # direction * [ r1 - beta*r2 ] / (1+|beta|), the same convention on both legs
            r1 = pair_prices[ticker1].iloc[day] / entryPrice1 - 1.0
            r2 = pair_prices[ticker2].iloc[day] / entryPrice2 - 1.0
            return direction * ((r1 - beta * r2) / (1 + abs(beta))) * 100

        tradeReturn = markedReturn(exitDay)

        exitReasons.setdefault(exitReason, []).append(tradeReturn) # files every trade's return under the reason it closed

        if direction == -1:
            totalProfitPos.append(tradeReturn)
        else:
            totalProfitNeg.append(tradeReturn)

        cumulative = 0.0

        for d in range(entryDay + 1, exitDay + 1): # walks through each day the trade was open to record its daily change
            newCumulative = markedReturn(d)
            dailyPnL.iloc[d] = newCumulative - cumulative # only the change since yesterday, not the running total
            cumulative = newCumulative

    meanProfitPos = np.mean(totalProfitPos) if totalProfitPos else 0.0
    meanProfitNeg = np.mean(totalProfitNeg) if totalProfitNeg else 0.0

    return float(meanProfitPos), float(meanProfitNeg), len(totalProfitPos), len(totalProfitNeg), deathDay, len(pair_prices), dailyPnL, totalProfitPos + totalProfitNeg, exitReasons

if __name__ == "__main__": # only runs when this file is executed directly, not when it's imported
    downloadStart = (pd.to_datetime(start_date) - pd.DateOffset(years=recheckLookbackYears)).strftime("%Y-%m-%d") # pulls extra history so the first recheck has a full lookback behind it

    from PriceDataAndADF import screenPairs

    records = screenPairs("2018-08-20", "2020-08-20")

    allPrices = pd.read_pickle("prices.pkl")

    for record in records[:3]:
        t1, t2 = record["ticker1"], record["ticker2"]
        pair_prices = allPrices.loc["2018-08-20":"2021-08-20", [t1, t2]].dropna()
        print(t1, t2, profitTesting(t1, t2, pair_prices, "2020-08-20", record)[:4])



    



