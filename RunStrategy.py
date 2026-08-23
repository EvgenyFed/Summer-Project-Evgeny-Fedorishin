import pandas as pd
import numpy as np
import os

from ProfitTesting import profitTesting, recheckLookbackYears
from PriceDataAndADF import screenPairs

folder = os.path.dirname(os.path.abspath(__file__))

screenYears = 2   # length of each screening window
tradeYears = 1    # length of each trading window
folds = 8         # how many times we screen then trade, rolling forward through history

allPrices = pd.read_pickle(os.path.join(folder, "prices.pkl"))

today = pd.Timestamp.today()

foldSummaries = []
allPortfolioDaily = [] # each fold's daily portfolio moves, stitched together at the end
allTrades = []

for f in range(folds):
    screenStart = (today - pd.DateOffset(years=screenYears + tradeYears + (folds - 1 - f))).strftime("%Y-%m-%d")
    screenEnd = (today - pd.DateOffset(years=tradeYears + (folds - 1 - f))).strftime("%Y-%m-%d")
    tradeEnd = (today - pd.DateOffset(years=(folds - 1 - f))).strftime("%Y-%m-%d")

    pairs = screenPairs(screenStart, screenEnd) # picks fresh pairs using only data before the trading period

    print("fold", f + 1, "screened", screenStart, "to", screenEnd, "- trading", screenEnd, "to", tradeEnd, "-", len(pairs), "pairs")

    leadInStart = (pd.to_datetime(screenEnd) - pd.DateOffset(years=recheckLookbackYears)).strftime("%Y-%m-%d") # extra history so the rolling window and first recheck have data behind them

    foldCurves = {}

    for ticker1, ticker2 in pairs:
        pair_prices = allPrices.loc[leadInStart:tradeEnd, [ticker1, ticker2]].dropna()

        meanProfitPos, meanProfitNeg, tradesPos, tradesNeg, deathDay, totalDays, dailyPnL, tradeProfits = profitTesting(ticker1, ticker2, pair_prices, screenEnd)

        foldCurves[ticker1 + "/" + ticker2] = dailyPnL.loc[screenEnd:tradeEnd] # cuts off the lead-in so folds don't overlap when stitched

        allTrades.extend(tradeProfits)

    foldTable = pd.DataFrame(foldCurves).fillna(0.0)

    foldDaily = foldTable.mean(axis=1) # equal capital per pair so the portfolio move is the average

    allPortfolioDaily.append(foldDaily)

    foldSummaries.append({
        "fold": f + 1,
        "tradingFrom": screenEnd,
        "tradingTo": tradeEnd,
        "pairs": len(pairs),
        "return": ((1 + foldDaily / 100).prod() - 1) * 100, # compounded within the fold to match the overall equity curve
    })

portfolioDaily = pd.concat(allPortfolioDaily) # one continuous series across all eight trading years

portfolioDaily = portfolioDaily[~portfolioDaily.index.duplicated(keep="first")] # guards against a shared boundary date appearing twice

equity = 100.0 * (1 + portfolioDaily / 100).cumprod() # compounds each day's percentage rather than just adding them up

runningPeak = equity.cummax()

drawdowns = (equity - runningPeak) / runningPeak * 100

maxDrawdown = drawdowns.min()

dailyVol = portfolioDaily.std()

annualVol = dailyVol * np.sqrt(252)

years = len(portfolioDaily) / 252

annualReturn = ((equity.iloc[-1] / 100.0) ** (1 / years) - 1) * 100 # the compound annual growth rate rather than a straight-line average

sharpe = annualReturn / annualVol if annualVol > 0 else 0.0

wins = len([t for t in allTrades if t > 0])

winRate = wins / len(allTrades) * 100 if allTrades else 0.0

print()
print(pd.DataFrame(foldSummaries).to_string(index=False))

print()
print("final equity:", round(equity.iloc[-1], 3))
print("total return:", f"{equity.iloc[-1] - 100.0:+.3f}%")
print("annualised return:", f"{annualReturn:+.3f}%")
print("max drawdown:", f"{maxDrawdown:+.3f}%")
print("deepest below peak on:", drawdowns.idxmin().strftime("%Y-%m-%d"))
print("worst single day:", f"{portfolioDaily.min():+.3f}%", "on", portfolioDaily.idxmin().strftime("%Y-%m-%d"))
print("best single day:", f"{portfolioDaily.max():+.3f}%", "on", portfolioDaily.idxmax().strftime("%Y-%m-%d"))
print("annualised volatility:", f"{annualVol:.3f}%")
print("sharpe ratio:", round(sharpe, 3))
print("win rate:", f"{winRate:.1f}%", "(" + str(wins) + " of " + str(len(allTrades)) + " trades)")