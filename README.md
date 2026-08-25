Strategy

A rolling walk-forward pairs trading backtest over 10 years of S&P 500 data.

Screen:  within each GICS sector, test every stock pair for co-integration (ADF) over a 2-year formation window; rank by p-value and keep the top 10. Restricting to same-sector pairs targets relationships with an economic reason to mean-revert, rather than the spurious pairs you get from testing all ~125,000 combinations.
Trade:  run the strategy on the following year, entering on z-score divergence and exiting on reversion.
Retest:  every 2 months, re-run co-integration on the trailing 2 years and drop pairs that fail
.
Roll:  step the window forward and repeat across the full 10 years.

<img width="1417" height="904" alt="equity_curve" src="https://github.com/user-attachments/assets/bfd54f36-1a41-4c2c-b72a-fa4ac7b6c948" />
