# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 09:12:02 2023

@author: banik
"""

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
from itertools import chain
# plt.style.use('fast')
from matplotlib import cm
from matplotlib.ticker import LinearLocator


asset = yf.Ticker("SPY")
expirations = asset.options

def option_chains(ticker):
    """
    """
    chains = pd.DataFrame()
    
    for expiration in expirations:
        # tuple of two dataframes
        opt = asset.option_chain(expiration)
        
        calls = opt.calls
        calls['optionType'] = "call"
        
        puts = opt.puts
        puts['optionType'] = "put"
        
        chain = pd.concat([calls, puts])
        chain['expiration'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)
        
        chains = pd.concat([chains, chain])
    
    chains["daysToExpiration"] = (chains.expiration - dt.datetime.today()).dt.days + 1
    
    return chains
    # return calls

# store maturities
lMaturity = list(asset.options)

# get current date
today = datetime.now().date()
# empty list for days to expiration
DTE = []
# empty list to store data for calls
Call_list = []
# empty list to store data for puts
Put_list = []

# loop over maturities
for maturity in lMaturity:
    # maturity date
    maturity_date = datetime.strptime(maturity, '%Y-%m-%d').date()
    # DTE: difference between maturity date and today
    DTE.append((maturity_date - today).days)
    # store call data
    Call_list.append(asset.option_chain(maturity).calls)
    Put_list.append(asset.option_chain(maturity).puts)


#-----------------------------------------------------------------------------#
#Analyze skew and term structure for Call Options
#-----------------------------------------------------------------------------#
options = option_chains("SPY")
pd.set_option('display.max_columns', None)
print(options)
calls = options[options["optionType"] == "call"]
calls.expiration = pd.to_datetime(calls.expiration).dt.date

# Next, pick an expiration so you can plot the volatility skew.
# print the expirations
set(calls.expiration)

# select an expiration to plot
calls_at_expiry = calls[calls["expiration"] ==  calls.expiration.unique()[1]]#"2023-03-20 23:59:59"]

# filter out low vols
filtered_calls_at_expiry = calls_at_expiry[calls_at_expiry.impliedVolatility >= 0.001].head(-1)

# set the strike as the index so pandas plots nicely
filtered_calls_at_expiry[["strike", "impliedVolatility"]].set_index("strike").plot(
    title="Implied Volatility Skew of call options", figsize=(7, 4)
)


# select an expiration to plot
calls_at_strike = calls[calls["strike"] == 400.0]

# filter out low vols
filtered_calls_at_strike = calls_at_strike[calls_at_strike.impliedVolatility >= 0.001].head(-1)

# set the strike as the index so pandas plots nicely
filtered_calls_at_strike[["expiration", "impliedVolatility"]].set_index("expiration").plot(
    title="Implied Volatility Term Structure of call options", figsize=(7, 4)
)

#-----------------------------------------------------------------------------#
#Analyze skew and term structure for Put Options
#-----------------------------------------------------------------------------#
puts = options[options["optionType"] == "put"]
puts.expiration = pd.to_datetime(puts.expiration).dt.date

# Next, pick an expiration so you can plot the volatility skew.
# print the expirations
set(puts.expiration)

# select an expiration to plot
puts_at_expiry = puts[puts["expiration"] ==  puts.expiration.unique()[1]]#"2023-03-20 23:59:59"]

# filter out low vols
filtered_puts_at_expiry = puts_at_expiry[puts_at_expiry.impliedVolatility >= 0.001].head(-1)

# set the strike as the index so pandas plots nicely
filtered_puts_at_expiry[["strike", "impliedVolatility"]].set_index("strike").plot(
    title="Implied Volatility Skew of put options", figsize=(7, 4)
)


# select an expiration to plot
puts_at_strike = puts[puts["strike"] == 400.0]

# filter out low vols
filtered_puts_at_strike = puts_at_strike[puts_at_strike.impliedVolatility >= 0.001].head(-1)

# set the strike as the index so pandas plots nicely
filtered_puts_at_strike[["expiration", "impliedVolatility"]].set_index("expiration").plot(
    title="Implied Volatility Term Structure of put options", figsize=(7, 4)
)

#-----------------------------------------------------------------------------#
# PLOT A IMPLIED VOLATILITY SURFACE for Calls
#-----------------------------------------------------------------------------#
# pivot the dataframe
call_surface = (
    calls[['daysToExpiration', 'strike', 'impliedVolatility']]
    .pivot_table(values='impliedVolatility', index='strike', columns='daysToExpiration')
    .dropna()
)

# create the figure object
fig = plt.figure(figsize=(10, 8))

# add the subplot with projection argument
ax = fig.add_subplot(111, projection='3d')

# get the 1d values from the pivoted dataframe
x, y, z = call_surface.columns.values, call_surface.index.values, call_surface.values

# return coordinate matrices from coordinate vectors
X, Y = np.meshgrid(x, y)

# set labels
ax.set_xlabel('Days to expiration')
ax.set_ylabel('Strike price')
ax.set_zlabel('Implied volatility')
ax.set_title('Implied volatility surface for Call Options',color = 'black',fontsize = 20)

# plot
surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

#-----------------------------------------------------------------------------#
#PLOT A IMPLIED VOLATILITY SURFACE for Puts
#-----------------------------------------------------------------------------#
# pivot the dataframe
put_surface = (
    puts[['daysToExpiration', 'strike', 'impliedVolatility']]
    .pivot_table(values='impliedVolatility', index='strike', columns='daysToExpiration')
    .dropna()
)

# create the figure object
fig = plt.figure(figsize=(10, 8))

# add the subplot with projection argument
ax = fig.add_subplot(111, projection='3d')

# get the 1d values from the pivoted dataframe
x, y, z = put_surface.columns.values, put_surface.index.values, put_surface.values

# return coordinate matrices from coordinate vectors
X, Y = np.meshgrid(x, y)

# set labels
ax.set_xlabel('Days to expiration')
ax.set_ylabel('Strike price')
ax.set_zlabel('Implied volatility')
ax.set_title('Implied volatility surface for Put Options',color = 'black',fontsize = 20)

# plot
surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


#-----------------------------------------------------------------------------#
#Fitting Spline for Call Options
#-----------------------------------------------------------------------------#
from scipy.interpolate import InterpolatedUnivariateSpline
# Get the implied volatilities for a specific maturity date
for i in range(0,len(Call_list)):
    maturity_date =  Call_list[i]
    ivs = Call_list[i]["impliedVolatility"]

# Define the strike prices and fit a spline model to the implied volatilities
    strikes = Call_list[i]["strike"]
    spline = InterpolatedUnivariateSpline(strikes, ivs)

# Plot the fitted spline
    fig, ax = plt.subplots()
    ax.plot(strikes, ivs, 'o', label='data')
    ax.plot(strikes, spline(strikes), label='spline')
    ax.legend()
    ax.set_xlabel("Strike price")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f'Call Options expiring at {expirations[i]}',fontsize=10,color='black')
    plt.show()
    
#-----------------------------------------------------------------------------#
#Fitting Spline for Put Options
#-----------------------------------------------------------------------------#
# Get the implied volatilities for a specific maturity date
for i in range(0,len(Put_list)):
    maturity_date =  Put_list[i]
    ivs = Put_list[i]["impliedVolatility"]

# Define the strike prices and fit a spline model to the implied volatilities
    strikes = Put_list[i]["strike"]
    spline = InterpolatedUnivariateSpline(strikes, ivs)

# Plot the fitted spline
    fig, ax = plt.subplots()
    ax.plot(strikes, ivs, 'o', label='data')
    ax.plot(strikes, spline(strikes), label='spline')
    ax.legend()
    ax.set_xlabel("Strike price")
    ax.set_ylabel("Implied Volatility")
    ax.set_title(f'Put Options expiring at {expirations[i]}',fontsize=10,color='black')
    plt.show()

#-----------------------------------------------------------------------------#
# Risk neutral density for Call Options
#-----------------------------------------------------------------------------#
import scipy.stats as stats
from scipy.stats import norm
sm_calls = calls.loc[calls['expiration'] == calls.expiration.unique()[1]]

S0 = float(asset.history("1d","1d").Close) # current price
T = sm_calls.daysToExpiration[1]/252# time to maturity (in years)
sigma = sm_calls.impliedVolatility[1]*np.sqrt(T) # volatility
K = sm_calls.strike[1] # strike price
r = 0.02 # risk-free interest rate
drift = r - 0.5 * sigma**2
x = np.linspace(330, 370, 1000)
d1 = (np.log(x / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
cdf = norm.cdf(d1)
pdf = (1 / (x * sigma * np.sqrt(T))) * norm.pdf(d1)
d2 = d1 - sigma * np.sqrt(T)
delta = norm.cdf(d1)
vega = x * norm.pdf(d1) * np.sqrt(T)
density = pdf * (delta + vega * (drift / sigma))

plt.plot(x, density)
plt.xlabel('Price')
plt.ylabel('Density')
plt.title(f'Risk-neutral density for the Call Option maturing at {expirations[1]}')
plt.show()


#-----------------------------------------------------------------------------#
# Risk neutral density for Put Options
#-----------------------------------------------------------------------------#
sm_puts = puts.loc[puts['expiration'] == puts.expiration.unique()[1]]

S = float(asset.history("1d","1d").Close) # current price
T = sm_puts.daysToExpiration[1]/252# time to maturity (in years)
sigma = sm_puts.impliedVolatility[1]*np.sqrt(T)#/np.sqrt(252) # volatility
K = sm_puts.strike[1] # strike price
r = 0.02 # risk-free interest rate
drift = r - 0.5 * sigma**2
x = np.linspace(300, 340, 1000)
d1 = (np.log(x / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
cdf = norm.cdf(d1)
pdf = (1 / (x * sigma * np.sqrt(T))) * norm.pdf(d1)
d2 = d1 - sigma * np.sqrt(T)
delta = norm.cdf(d1)
vega = x * norm.pdf(d1) * np.sqrt(T)
density = pdf * (delta + vega * (drift / sigma))

plt.plot(x, density)
plt.xlabel('Price')
plt.ylabel('Density')
plt.title(f'Risk-neutral density for the Put Option maturing at {expirations[1]}')
plt.show()

#-----------------------------------------------------------------------------#
# Implied volatility of call option using Black-Scholes Model
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.stats import norm

N_prime = norm.pdf
N = norm.cdf

def black_scholes_call(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: call price
    '''

    ###standard black-scholes formula
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * N(d1) -  N(d2)* K * np.exp(-r * T)
    return call

volatility_candidates = np.arange(0.01,4,0.0001)
price_differences = np.zeros_like(volatility_candidates)

call_price = sm_calls.lastPrice[3]
S = float(asset.history("1d","1d").Close)
K = sm_calls.strike[3]
r = 0.02 
T = sm_calls.daysToExpiration[3]/252 

for i in range(len(volatility_candidates)):
    
    candidate = volatility_candidates[i]
    
    price_differences[i] = call_price - black_scholes_call(S, K , T, r, candidate)
    
    
idx = np.argmin(abs(price_differences))
implied_volatility_call = volatility_candidates[idx]
print('Fitted volatility for call option is:', round(implied_volatility_call,3))
print("")
#-----------------------------------------------------------------------------#
# Implied volatility of put option using Black-Scholes Model
#-----------------------------------------------------------------------------#
import numpy as np
from scipy.stats import norm

N_prime = norm.pdf
N = norm.cdf

def black_scholes_put(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: put price
    '''

    ###standard black-scholes formula
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put = -  N(-d2)* K * np.exp(-r * T) - S * N(-d1)
    return put

volatility_candidates = np.arange(0.01,4,0.0001)
price_differences = np.zeros_like(volatility_candidates)

put_price = sm_puts.lastPrice[3]
S = float(asset.history("1d","1d").Close)
K = sm_puts.strike[3]
r = 0.02 
T = sm_puts.daysToExpiration[3]/252 

for i in range(len(volatility_candidates)):
    
    candidate = volatility_candidates[i]
    
    price_differences[i] = put_price - black_scholes_put(S, K , T, r, candidate)
    
    
idx = np.argmin(abs(price_differences))
implied_volatility_put = volatility_candidates[idx]
print('Fitted volatility for put option is:', round(implied_volatility_put,3))
print("")
#-----------------------------------------------------------------------------#
# Writing a trading program that sells over-valued call options 
#and buys undervalued call options 
#-----------------------------------------------------------------------------#
# BUY condition
sm_calls['Trade_signal'] = np.where((sm_calls['impliedVolatility'] < implied_volatility_call),1,0)

# SELL condition
sm_calls['Trade_signal'] = np.where((sm_calls['impliedVolatility'] > implied_volatility_call),-1,sm_calls['Trade_signal'])

# creating long and short positions 
sm_calls['Trade_position'] = np.where((sm_calls['Trade_signal'] == 1),'BUY','SELL')

#-----------------------------------------------------------------------------#
# Writing a trading program that sells over-valued put options 
#and buys undervalued put options 
#-----------------------------------------------------------------------------#
# BUY condition
sm_puts['Trade_signal'] = np.where((sm_puts['impliedVolatility'] < implied_volatility_put),1,0)

# SELL condition
sm_puts['Trade_signal'] = np.where((sm_puts['impliedVolatility'] > implied_volatility_put),-1,sm_puts['Trade_signal'])

# creating long and short positions 
sm_puts['Trade_position'] = np.where((sm_puts['Trade_signal'] == 1),'BUY','SELL')



print(f'Trade signal for call option with expiration at {expirations[1]}')
print("")
print(sm_calls.iloc[:,-4:])
print("/n")
print(f'Trade signal for put option with expiration at {expirations[1]}')
print("")
print(sm_puts.iloc[:,-4:])
