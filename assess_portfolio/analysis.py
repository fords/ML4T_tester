"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

def compute_daily_returns(df):
    """ Compute and return the daily return values"""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:]/ df[:-1].values) -1
	#daily_return = daily_retuns[1:]
	#for each in daily_return:
	#	print each
    # daily_returns.ix[0,0] = 0
    return daily_returns

def compute_cumu_returns(df):
    """ Compute and return the daily cumulative return values"""
    cumulative_df = df.copy()
    cumulative_df[0:] = (df[0:]/ df.ix[0,:]) -1
    return cumulative_df


# This is the function that will be tested by the autograder
# update this code to properly implement the functionality
def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    df = get_data(syms,dates)
    #plot_data(df)
    #gold= df.ix[0,4]   spy = df.ix[0,1]
    #print "testing",SPYa

    #print df
    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    normed = prices.copy()
    normed[0:] = (prices[0:]/ prices.ix[0,:])
    alloced = normed * allocs
    pos_vals = alloced * sv
    port_val = pos_vals.sum(axis = 1)
    print port_val.ix[1,0]
    #print port_val
    #port_val = prices_SPY # add code here to compute daily portfolio values


    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    #cr = compute_cumu_returns(port_val)
    cr = (port_val.ix[-1,:]/port_val.ix[0,:])  -1
    daily_rets = []
    daily_rets = compute_daily_returns(port_val)
    daily_rets = daily_rets[1:]         # daily returns

    #if sf == 252:
    #    #average daily returns mean
    #adr = daily_rets
    #else:
    adr = daily_rets.mean()

    sddr = daily_rets.std()   # standard deviation daily return
    difference = sddr - rfr
    mean = difference.mean()
    sr =  np.sqrt(sf)* mean/sddr                  #sharpe ratio

    # Compare daily portfolio value with SPY using a normalized plot

    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_data( df_temp, title= "Plot" , ylabel= "Prices")
        pass

    # Add code here to properly compute end value
    #dates = pd.date_range(ed)  wrong
    #port_evdata = get_data(port_val,dates)
    ev = port_val.ix[-1,:]
    #print ("ev",ev)
    return cr, adr, sddr, sr, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,1,1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, \
        gen_plot = True)

    # Print statistics
    """print ("Start Date:", start_date)
    print ("End Date:", end_date)
    print ("Symbols:", symbols)
    print ("Allocations:", allocations)
    print ("Sharpe Ratio:", sr)
    print ("Volatility (stdev of daily returns):", sddr)
    print ("Average Daily Return:", adr)
    print ("Cumulative Return:", cr)"""

    print "testing"
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    test_code()
