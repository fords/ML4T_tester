"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

def author():
    return 'zwin3'

def compute_portvals(orders, start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
    portvals = portvals[['IBM']]  # remove SPY
    rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())

    orders_df = orders
    orders_df.sort_index(inplace=True)
    dates = pd.date_range(min(orders_df.index), max(orders_df.index))
    symbols = []  # array
    symbols = list(set(orders_df['Symbol']))
    prices = get_data(symbols, dates)


    cash_df = pd.DataFrame( index = dates, columns = ['Cash'])
    cash_df = cash_df.fillna(1.000)
    prices = prices.join(cash_df)
    trades = pd.DataFrame(.0, columns = prices.columns, index = prices.index)
    comission_df = pd.DataFrame(index = prices.index, columns = ['Commission'])
    comission_df = comission_df.fillna(.00)
    impact_df = pd.DataFrame(index = prices.index, columns = ['Impact'])
    impact_df = impact_df.fillna(.00)

    for i, iterrows in orders_df.iterrows():
        shares = iterrows['Shares']
        symbols = iterrows['Symbol']
        if (iterrows['Order'] == 'SELL'):
            trades.loc[i][symbols] = trades.loc[i][symbols] + (-1 * shares)
        elif (iterrows['Order'] == 'BUY'):
            trades.loc[i][symbols] = trades.loc[i][symbols] + (1 * shares)
        comission_df.loc[i]['Commission'] = comission_df.loc[i]['Commission'] + commission
        impact_df.loc[i]['Impact'] = impact_df.loc[i]['Impact'] + (prices.loc[i][symbols] * shares * impact)

    temp_df = prices * trades
    trades['Cash'] = -1.0 * temp_df.sum(axis = 1)
    trades['Cash'] = trades['Cash'] - comission_df['Commission'] - impact_df['Impact']
    holdings = pd.DataFrame( .0, columns = trades.columns, index = trades.index)
    holdings.loc[min(trades.index), 'Cash'] = start_val   # start_date = min(trades.index)
    holdings = holdings + trades
    portvals = (prices * holdings.cumsum()).sum(axis = 1)

    #return rv
    return portvals

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

if __name__ == "__main__":
    start_date= dt.datetime(2011,01,14)
    end_date = dt.datetime(2011,12,14)
    of= "./orders/orders2.csv"
    test_code(start_date,end_date, of)
    #test_code(dt.datetime(2011,01,10), dt.datetime(2011,12,20), "./orders/orders.csv")
