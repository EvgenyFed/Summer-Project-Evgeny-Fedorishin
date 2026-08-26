Strategy

A rolling walk-forward pairs trading backtest over 10 years of S&P 500 data.

SCREEN:  within each GICS sector, test every stock pair for co-integration (ADF) over a 2-year formation window; rank by p-value and keep the top 10.
TRADE:  run the strategy on the following year, entering on z-score divergence and exiting on reversion.
RETEST:  every 2 months, re-run co-integration on the trailing 2 years and drop pairs that fail
ROLL:  step the window forward and repeat across the full 10 years.

Over an 8-fold walk-forward, the strategy returned 7% cumulative (0.24 Sharpe, 10.2% max drawdown). This is consistent with published findings that classical pairs trading returns have decayed substantially since the mid-2000s. No parameters were tuned on the test period and every pair selection used only data preceding the trading window.

<img width="1417" height="904" alt="equity_curve" src="https://github.com/user-attachments/assets/bfd54f36-1a41-4c2c-b72a-fa4ac7b6c948" />
