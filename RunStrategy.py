import yfinance as yf
import pandas as pd
import numpy as np

from ProfitTesting import profitTesting, start_date, end_date, recheckLookbackYears

pairs = [ # hardcoded for now, will come from PriceDataAndADF later
    ("EG", "V"),
    ("MSCI", "PYPL"),
    ("PNR", "VRT"),
    ("KEY", "RF"),
    ("EG", "MA"),
    ("EG", "JPM"),
    ("EG", "HOOD"),
    ("FDX", "JBHT"),
    ("LYV", "NWS"),
    ("REG", "SBAC"),
]

downloadStart = (pd.to_datetime(start_date) - pd.DateOffset(years=recheckLookbackYears)).strftime("%Y-%m-%d") # same lead-in the backtest needs for its rechecks

allTickers = sorted(set([t for pair in pairs for t in pair])) # every ticker across every pair, deduplicated

prices = yf.download(allTickers, start=downloadStart, end=end_date, auto_adjust=True)["Close"] # one download for all pairs instead of one per pair

results = []
allCurves = {} # each pair's daily pnl series, collected so they can be combined into one portfolio

for ticker1, ticker2 in pairs:
    pair_prices = prices[[ticker1, ticker2]].dropna() # slices the shared download down to this pair

    meanProfitPos, meanProfitNeg, tradesPos, tradesNeg, deathDay, totalDays, dailyPnL = profitTesting(ticker1, ticker2, pair_prices, start_date)

    results.append({
        "pair": ticker1 + "/" + ticker2,
        "meanProfitPos": meanProfitPos,
        "meanProfitNeg": meanProfitNeg,
        "tradesPos": tradesPos,
        "tradesNeg": tradesNeg,
        "died": "no" if deathDay == totalDays else "day " + str(deathDay),
        "sumDailyPnL": dailyPnL.sum(), # should match the total of all trade profits if the daily curve is right
        "activeDays": int((dailyPnL != 0).sum()), # how many days the pair actually had a position open
    })

    allCurves[ticker1 + "/" + ticker2] = dailyPnL

resultsTable = pd.DataFrame(results)

print(resultsTable.to_string(index=False))

curvesTable = pd.DataFrame(allCurves).fillna(0.0) # lines every pair up on shared dates, 0 on days a pair has no data or no open trade

portfolioDaily = curvesTable.mean(axis=1) # each pair gets an equal share of capital so the portfolio's daily move is the average of the pairs

equity = 100.0 + portfolioDaily.cumsum() # account value starting from 100

runningPeak = equity.cummax() # the highest the account has been up to each day

drawdowns = (equity - runningPeak) / runningPeak * 100 # how far below the peak we are on each day, as a percentage

maxDrawdown = drawdowns.min()

print()
print("final equity:", round(equity.iloc[-1], 3))
print("total return %:", round(equity.iloc[-1] - 100.0, 3))
print("max drawdown %:", round(maxDrawdown, 3))
print("worst day:", drawdowns.idxmin().strftime("%Y-%m-%d"))