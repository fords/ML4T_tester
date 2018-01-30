"""MC1-P2: Optimize a portfolio.

"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from scipy.optimize import minimize, rosen, rosen_der
from util import get_data, plot_data


def compute_daily_returns(df):
    """ Compute and return the daily return values"""
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:]/ df[:-1].values) -1
    return daily_returns

def compute_cumu_returns(df):
    """ Compute and return the daily cumulative return values"""
    cumulative_df = df.copy()
    cumulative_df[0:] = (df[0:]/ df.ix[0,:]) -1
    return cumulative_df


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
    #print "df",df.ix[:,1]   # 0 index first data 'GOOG'
    #plot_data(df)
    # Compute daily returns
    daily_returns = compute_daily_returns(df)
    normed = prices.copy()
    normed[0:] = (prices[0:]/ prices.ix[0,:])
    alloced = normed * allocs
    pos_vals = alloced * sv
    port_val = pos_vals.sum(axis = 1)
    #print port_val.ix[1,0]
    #print port_val
    #port_val = prices_SPY # add code here to compute daily portfolio values

    # Get portfolio statistics (note: std_daily_ret = volatility)
    #cr = compute_cumu_returns(port_val)
    cr = (port_val.ix[-1,:]/port_val.ix[0,:])  -1
    daily_rets = []
    daily_rets = compute_daily_returns(port_val)
    daily_rets = daily_rets[1:]         # daily returns
    adr = daily_rets.mean()
    sddr = daily_rets.std()   # standard deviation daily return
    difference = daily_rets - rfr
    mean_val = difference.mean()
    sr =  np.sqrt(sf)* mean_val/sddr                  #sharpe ratio

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_data( df_temp, title= "Plot" , ylabel= "Prices")
        pass

    ev = port_val.ix[-1,:]
    #print ("ev",ev)
    return cr, adr, sddr, sr, ev, port_val

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):
    sv  = 1
    rfr = 0
    sf = 252
    # find allocations values
    no_syms = len(syms)
    one = 1.0
    allocations = []
    allocation_no = one/no_syms
    for i in range(0,no_syms):
        allocations.append(allocation_no)
    #print allocations
    #allocations = np.asarray([0.2, 0.2, 0.3, 0.3])

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices_nor_SPY = prices_SPY / prices_SPY.ix[0,:]
    normed = prices.copy()
    normed[0:] = (prices[0:]/ prices.ix[0,:])
    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    def f(allocations_final):
        return (compute_daily_returns((normed*allocations_final).sum(axis = 1)))[1:] .std() - 1

    def constraint(allocations_final):
        sum = 0
        for each in allocations_final:
            sum += each
        return 1-sum
    bnds = ()
    bnds_list = []
    b =(0,1)
    for i in range(0,no_syms):
        bnds_list.append(b)
    bnds = tuple(bnds_list)
    con1 = {'type': 'eq', 'fun': constraint} #, bounds=(0.0,1.0)
    res = minimize(f,allocations, method = 'SLSQP',bounds=bnds,constraints = con1)
    #print res
    #allocs = np.asarray([0.2, 0.2, 0.3, 0.3]) # add code here to find the allocations
    cr, adr, sddr, sr, ev, port_val = assess_portfolio(sd = sd, ed = ed,\
        syms = syms, \
        allocs = res.x,\
        sv = 1, \
        gen_plot = False)
    #cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    # add to bfill and ffill

    # Get daily portfolio value
    # add code here to compute daily portfolio values

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_nor_SPY], keys=['Portfolio', 'SPY'], axis=1)
        #pass
        plot_data( df_temp, title= "Optimal portfolio with SPY" , ylabel= "Prices")

    return res.x, cr, adr, sddr, sr

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
