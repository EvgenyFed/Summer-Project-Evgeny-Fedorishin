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

downloadStart = (pd.to_datetime(start_date) - pd.DateOffset(years=recheckLookbackYears)).strftime("%Y-%m-%d")

allTickers = sorted(set([t for pair in pairs for t in pair])) # every ticker across every pair, deduplicated

prices = yf.download(allTickers, start=downloadStart, end=end_date, auto_adjust=True)["Close"] # one download for all pairs instead of one per pair

results = []
allCurves = {} # each pair's daily pnl series, collected so they can be combined into one portfolio
allTrades = [] # every individual trade profit across every pair, for the win rate

for ticker1, ticker2 in pairs:
    pair_prices = prices[[ticker1, ticker2]].dropna() # slices the shared download down to this pair

    meanProfitPos, meanProfitNeg, tradesPos, tradesNeg, deathDay, totalDays, dailyPnL, tradeProfits = profitTesting(ticker1, ticker2, pair_prices, start_date)

    results.append({
        "pair": ticker1 + "/" + ticker2,
        "meanProfitPos": meanProfitPos,
        "meanProfitNeg": meanProfitNeg,
        "tradesPos": tradesPos,
        "tradesNeg": tradesNeg,
        "died": "no" if deathDay == totalDays else "day " + str(deathDay),
        "sumDailyPnL": dailyPnL.sum(), 
        "activeDays": int((dailyPnL != 0).sum()), # how many days the pair actually had a position open
    })

    allCurves[ticker1 + "/" + ticker2] = dailyPnL

    allTrades.extend(tradeProfits)

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
print("total return:", f"{equity.iloc[-1] - 100.0:+.3f}%")
print("max drawdown:", f"{maxDrawdown:+.3f}%")
print("deepest below peak on:", drawdowns.idxmin().strftime("%Y-%m-%d"))
print("worst single day:", f"{portfolioDaily.min():+.3f}%", "on", portfolioDaily.idxmin().strftime("%Y-%m-%d"))
print("best single day:", f"{portfolioDaily.max():+.3f}%", "on", portfolioDaily.idxmax().strftime("%Y-%m-%d"))

dailyVol = portfolioDaily.std() # how much the account's daily move typically varies

annualVol = dailyVol * np.sqrt(252) # scales the volatility to a yearly figure in order to later calculate the sharpe ratio 

tradingDays = len(portfolioDaily)

annualReturn = (equity.iloc[-1] - 100.0) * (252 / tradingDays) # scales the return to a yearly figure in order to later calculate the sharpe ratio

sharpe = annualReturn / annualVol if annualVol > 0 else 0.0 # return earned per unit of volatility

wins = len([t for t in allTrades if t > 0])

winRate = wins / len(allTrades) * 100 if allTrades else 0.0

print("annualised volatility:", str(round(annualVol, 3)) + "%")
print("sharpe ratio:", round(sharpe, 3))
print("win rate:", str(round(winRate, 1)) + "%", "(" + str(wins) + " of " + str(len(allTrades)) + " trades)")