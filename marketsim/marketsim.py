"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def author():
    return 'zwin3'

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
    portvals = portvals[['IBM']]  # remove SPY
    rv = pd.DataFrame(index=portvals.index, data=portvals.as_matrix())

    df_orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    df_orders.sort_index(inplace=True)

    dates = pd.date_range(df_orders.first_valid_index(), df_orders.last_valid_index())

    syms = np.array(df_orders.Symbol.unique()).tolist()
    df_prices = get_data(syms, dates)
    columns = ['Cash']
    df_cash = pd.DataFrame(index=dates, columns=columns)
    df_cash = df_cash.fillna(1.0)
    df_prices = df_prices.join(df_cash)


    df_trades = create_trades(df_orders, df_prices, commission, impact)

    df_holdings = create_holdings(df_trades, start_val)

    df_values = create_values(df_prices, df_holdings)

    df_portval = cal_portval(df_values)

    #return rv
    return df_portval
def create_trades(df_orders, df_prices, commission, impact):
    df_trades = pd.DataFrame(0.0, columns=df_prices.columns, index=df_prices.index)
    columns = ['Commission']
    df_commission = pd.DataFrame(index=df_prices.index, columns=columns)
    df_commission = df_commission.fillna(0.0)
    columns_2 = ['Impact']
    df_impact = pd.DataFrame(index=df_prices.index, columns=columns_2)
    df_impact = df_impact.fillna(0.0)
    for index, row in df_orders.iterrows():
        sym = row['Symbol']
        shares = row['Shares']
        a = -1
        if (row['Order'] == 'BUY'):
            a = 1
        df_trades.loc[index][sym] = df_trades.loc[index][sym] + (a * shares)
        df_commission.loc[index]['Commission'] = df_commission.loc[index]['Commission'] + commission
        df_impact.loc[index]['Impact'] = df_impact.loc[index]['Impact'] + (df_prices.loc[index][sym] * shares * impact)

    df_temp = (df_prices * df_trades)

    df_trades['Cash'] = (-1.0 * df_temp.sum(axis = 1))

    df_trades['Cash'] = df_trades['Cash'] - df_commission['Commission'] - df_impact['Impact']

    return df_trades

def create_holdings(df_trades, start_val):
    start_date = df_trades.first_valid_index()
    df_holdings = pd.DataFrame(0.0, columns=df_trades.columns, index=df_trades.index)
    df_holdings.loc[start_date, 'Cash'] = start_val
    df_holdings = df_holdings + df_trades
    df_holdings = df_holdings.cumsum()
    return df_holdings

def create_values(df_prices, df_holdings):
    return df_prices * df_holdings

def cal_portval(df_values):
    return df_values.sum(axis = 1)

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


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2011,01,14)
    end_date = dt.datetime(2011,12,14)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio   = assess_portfolio(portvals, sf =245)
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]
    portvals_SPX = (prices_SPX/prices_SPX.ix[0,:]).sum(axis=1)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = assess_portfolio(portvals_SPX,sf =245)
    #cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    daily_returns = portvals[1:].values / portvals[:-1] - 1
    cum_ret = portvals[-1] / portvals[0] - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(245) * avg_daily_ret / std_daily_ret  #sf = 245

    # Compare portfolio against $SPX
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
    test_code()
