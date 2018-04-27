# Zeyar Win zwin3

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

def author():
    return 'zwin3'

def compute_portvals(orders_file, start_val = 100000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    if len(orders_file.any(1).nonzero()[0]) == 0:
        return(pd.DataFrame(data=100000, index=orders_file.index, columns=[orders_file.columns[0]]))

    dates = orders_file.index
    symbol = orders_file.columns[0]

    prices_data = get_data([symbol], pd.date_range(dates[0],dates[-1]))
    if symbol != 'SPY':
        prices_data = prices_data.drop('SPY', axis=1)

    df_prices = pd.DataFrame(prices_data)
    df_prices['cash'] = 1
    df_trades = orders_file.copy()

    df_holdings = df_trades.copy()

    for i in orders_file.index:
        if orders_file.ix[i,symbol] != 0: 
            total_cost = orders_file.loc[i, symbol] * df_prices.loc[i, symbol] 
            df_trades.loc[i, 'cash'] = -total_cost - abs(commission + total_cost * impact)
    df_trades.fillna(0, inplace=True)

    df_holdings.loc[dates[0],'cash'] = start_val + df_trades.loc[dates[0],'cash']
    df_holdings.iloc[0, :-1] = df_trades.iloc[0, :-1]

    for i in range(1, df_holdings.shape[0]):
        df_holdings.iloc[i, :] = df_trades.iloc[i, :] + df_holdings.iloc[i-1, :]

    df_value = df_holdings.multiply(df_prices)

    portvals = df_value.sum(axis=1)
	#return rv
    return(portvals)

def assess_portfolio(portvals, rfr=0.0, sf=245.0, \
    gen_plot=False):


    # Get portfolio statistics (note: std_daily_ret = volatility)
    #cr = compute_cumu_returns(port_val)
    cr = (portvals.ix[-1,:]/portvals.ix[0,:])  -1
    daily_rets = []
    daily_rets = compute_daily_returns(portvals)
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

    ev = portvals.ix[-1,:]
    #print ("ev",ev)
    return cr, adr, sddr, sr


def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:]/ df[:-1].values) -1
    daily_returns = daily_returns[1:]
    return daily_returns

def compute_cumu_returns(df):
    """ Compute and return the daily cumulative return values"""
    cumulative_df = df.copy()
    cumulative_df[0:] = (df[0:]/ df.ix[0,:]) -1
    return cumulative_df


def test_code(start_date= dt.datetime(2011,01,14), end_date = dt.datetime(2011,12,14), of= "./orders/orders2.csv" ):
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    #of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    #start_date = dt.datetime(2011,01,14)
    #end_date = dt.datetime(2011,12,14)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio   = assess_portfolio(portvals, sf =245)
    prices_SPX = get_data(['$SPX'] , pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]
    portvals_SPX = (prices_SPX/prices_SPX.ix[0,:]).sum(axis=1)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = assess_portfolio(portvals_SPX,sf =252)
    #cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    daily_returns = portvals[1:].values / portvals[:-1] - 1
    cum_ret = portvals[-1] / portvals[0] - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(245) * avg_daily_ret / std_daily_ret  #sf = 245

    daily_returns_SPY = portvals_SPX[1:].values / portvals_SPX[:-1] - 1
    cum_ret_SPY = portvals_SPX[-1] / portvals_SPX[0] - 1
    avg_daily_ret_SPY = daily_returns_SPY.mean()
    std_daily_ret_SPY = daily_returns_SPY.std()
    sharpe_ratio_SPY = np.sqrt(252) * avg_daily_ret_SPY / std_daily_ret_SPY  #sf = 252

    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])
