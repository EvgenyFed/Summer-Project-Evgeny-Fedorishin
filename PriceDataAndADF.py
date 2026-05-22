import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

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

tickers = fixed_tickers


end_date = datetime.today().strftime("%Y-%m-%d") 
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d") #makes start data exactly a year ago from now

prices = yf.download(tickers, start = start_date, end = end_date, auto_adjust= True)["Close"]

beta = {}

for sector in sectors: #used to cycle through sectors
    for j in range(len(sectors[sector])): #used to cycle through first tickers
        for k in range(j+1, len(sectors[sector])): #used to cycle through second tickers
            returns1 = []
            returns2 = []
            numeratorSum = 0
            denominatorSum = 0
            sumOfNumerator = 0
            sumOfDenominator = 0      
            for i in range(1, len(prices)): #For each day, calculate the percentage price change for both stocks and store them in two lists. 
                returns1.append((prices[sectors[sector][j]].iloc[i] - prices[sectors[sector][j]].iloc[i-1]) / prices[sectors[sector][j]].iloc[i-1]) #calculates percentage change between each day for first ticker
                returns2.append((prices[sectors[sector][k]].iloc[i] - prices[sectors[sector][k]].iloc[i-1]) / prices[sectors[sector][k]].iloc[i-1]) #calculates percentage change between each day for second ticker
            for z in range(len(returns1)):  #calculates hedgeratio between the tickers
                numeratorSum = (returns1[z] * returns2[z]) + numeratorSum
                denominatorSum = (returns1[z] ** 2) + denominatorSum 
            hedgeRatio = numeratorSum / denominatorSum            
            for z in range(0, len(returns1) - 1): #calculates Beta between the tickers
                spread = returns1[z] - hedgeRatio * returns2[z]
                spreadNextDay = returns1[z+1] - hedgeRatio * returns2[z+1]
                sumOfNumerator = (spread * (spreadNextDay - spread)) + sumOfNumerator
                sumOfDenominator = sumOfDenominator + (spread ** 2)
            beta[sectors[sector][j], sectors[sector][k]] = sumOfNumerator / sumOfDenominator




top10 = {}

def getBeta(x): #returns the absolute value of a pair's beta from the top10 dictionary
    return abs(top10[x])

for pair, value in beta.items(): #finds 10 largest beta values
    if len(top10) < 10:
        top10[pair] = value
    else:
        minPair = min(top10, key=getBeta)
        if abs(value) > abs(top10[minPair]):
            del top10[minPair]
            top10[pair] = value

for pair, value in top10.items(): #prints out the 10 highest beta values and their corresponding pairs
    print(pair, value)

            





