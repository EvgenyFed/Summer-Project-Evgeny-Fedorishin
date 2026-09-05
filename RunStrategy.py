import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import Config
from ProfitTesting import profitTesting
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
allExitReasons = {} # how many trades left for each reason, needed for the attribution table

for f in range(folds):
    screenStart = (today - pd.DateOffset(years=screenYears + tradeYears + (folds - 1 - f))).strftime("%Y-%m-%d")
    screenEnd = (today - pd.DateOffset(years=tradeYears + (folds - 1 - f))).strftime("%Y-%m-%d")
    tradeEnd = (today - pd.DateOffset(years=(folds - 1 - f))).strftime("%Y-%m-%d")

    records = screenPairs(screenStart, screenEnd) # picks fresh pairs using only data before the trading period, each with its frozen formation parameters

    print("fold", f + 1, "screened", screenStart, "to", screenEnd, "- trading", screenEnd, "to", tradeEnd, "-", len(records), "pairs")

    foldCurves = {}

    for record in records:
        ticker1, ticker2 = record["ticker1"], record["ticker2"]

        pair_prices = allPrices.loc[screenStart:tradeEnd, [ticker1, ticker2]].dropna() # the formation window is the lead-in now, so rolling mode has history and fixed mode just ignores it

        meanProfitPos, meanProfitNeg, tradesPos, tradesNeg, deathDay, totalDays, dailyPnL, tradeProfits, exitReasons = profitTesting(ticker1, ticker2, pair_prices, screenEnd, record)

        foldCurves[ticker1 + "/" + ticker2] = dailyPnL.loc[screenEnd:tradeEnd] # cuts off the lead-in so folds don't overlap when stitched

        allTrades.extend(tradeProfits)

        for reason, returns in exitReasons.items():
            allExitReasons.setdefault(reason, []).extend(returns)
    foldTable = pd.DataFrame(foldCurves).fillna(0.0)

    foldDaily = foldTable.mean(axis=1) # equal capital per pair so the portfolio move is the average

    allPortfolioDaily.append(foldDaily)

    foldSummaries.append({
        "fold": f + 1,
        "tradingFrom": screenEnd,
        "tradingTo": tradeEnd,
        "pairs": len(records),
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
print("run:", Config.runName, "| params:", Config.paramMode)
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
print()
print("exit reason breakdown:")

for reason in sorted(allExitReasons, key=lambda r: -len(allExitReasons[r])): # which exits the trades actually took, and what each one earned
    returns = allExitReasons[reason]
    share = len(returns) / len(allTrades) * 100 if allTrades else 0.0
    reasonWins = len([r for r in returns if r > 0]) / len(returns) * 100
    print(f"  {reason:12s} {len(returns):4d} trades ({share:4.1f}%)  avg {np.mean(returns):+7.3f}%  total {np.sum(returns):+8.2f}%  win {reasonWins:4.1f}%")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, #plots a visual equity curve
                               gridspec_kw={"height_ratios": [3, 1]})

ax1.plot(equity.index, equity.values, linewidth=1.2)
ax1.set_title(f"Pairs Trading — Walk-Forward Equity Curve ({folds} folds, {Config.runName})")
ax1.set_ylabel("Portfolio value (start = 100)")
ax1.grid(alpha=0.3)
ax1.text(0.02, 0.95,
         f"CAGR {annualReturn:.1f}%   Vol {annualVol:.1f}%   "
         f"Sharpe {sharpe:.2f}   MaxDD {maxDrawdown:.1f}%",
         transform=ax1.transAxes, va="top", fontsize=9)

ax2.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.4)
ax2.set_ylabel("Drawdown (%)")
ax2.grid(alpha=0.3)

fig.savefig(os.path.join(folder, "equity_curve_" + Config.runName + ".png"), dpi=150, bbox_inches="tight")