import yfinance as yf
import pandas as pd
import os

folder = os.path.dirname(os.path.abspath(__file__)) # the folder this script lives in, so files save next to it no matter where you run from

yearsOfHistory = 10 # how far back to pull prices

url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
table = pd.read_csv(url)

fixed_tickers = []
sectors = {}

for i in range(len(table)): #reformats tickers so that they can be used by yfinance
    ticker = table["Symbol"].iloc[i].replace(".", "-")
    sector = table["GICS Sector"].iloc[i]
    fixed_tickers.append(ticker)

    if sector not in sectors:
        sectors[sector] = []

    sectors[sector].append(ticker) #makes a list of all the sectors

end_date = (pd.Timestamp.today() - pd.DateOffset(days=1)).strftime("%Y-%m-%d") # stops at yesterday so today's still-forming bar doesn't change results between runs
start_date = (pd.Timestamp.today() - pd.DateOffset(years=yearsOfHistory)).strftime("%Y-%m-%d")

print("downloading", len(fixed_tickers), "tickers from", start_date, "to", end_date)

prices = yf.download(fixed_tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]

prices.to_pickle(os.path.join(folder, "prices.pkl")) # saves the price table to disk so the other scripts can read it instantly

pd.to_pickle(sectors, os.path.join(folder, "sectors.pkl")) # saves the sector groupings too so the screener doesn't refetch the constituents csv

print("saved prices.pkl:", prices.shape[0], "days x", prices.shape[1], "tickers")
print("saved sectors.pkl:", len(sectors), "sectors")