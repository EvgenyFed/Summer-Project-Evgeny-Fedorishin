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

for ticker1, ticker2 in pairs:
    pair_prices = prices[[ticker1, ticker2]].dropna() # slices the shared download down to this pair

    meanProfitPos, meanProfitNeg, tradesPos, tradesNeg, deathDay, totalDays = profitTesting(ticker1, ticker2, pair_prices, start_date)

    results.append({
        "pair": ticker1 + "/" + ticker2,
        "meanProfitPos": meanProfitPos,
        "meanProfitNeg": meanProfitNeg,
        "tradesPos": tradesPos,
        "tradesNeg": tradesNeg,
        "died": "no" if deathDay == totalDays else "day " + str(deathDay),
    })

resultsTable = pd.DataFrame(results)

print(resultsTable.to_string(index=False))