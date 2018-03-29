import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode as msim
from util import get_data, plot_data

def bb_value( vals, window_size):
    return (vals - vals.rolling(window_size).mean())/(2*vals.rolling(window_size).std())

def bollinger_bands(sma_val,std_val):
    return sma_val + std_val*2 , sma_val - std_val*2   # upper band , lower band

def momentum_value(vals,window_size):
    return ((vals/vals.shift(window_size)) - 1)

def sma_value( vals, window_size):
    return vals.rolling(window_size).mean()

def fill_miss_vals( df):
    df_fill = df.fillna(method ='ffill')
    df_fill = df.fillna(method = 'bfill')
    return df_fill

def normalize_df(prices):
    fill_miss_vals(prices)
    return prices / prices.ix[0, :]


def testPolicy(symbol = "JPM", start_date=dt.datetime(2008,1,1), end_date=dt.datetime(2009,12,31), sv = 100000):
    syms = [symbol]
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(syms, dates)
    prices_SPY = prices_all['SPY']
    prices_SPY = normalize_df(prices_SPY)
    prices = prices_all[syms]
    prices = normalize_df(prices)
    orders = pd.DataFrame(0, index = prices.index, columns = ['Shares'])
    buy_sell = pd.DataFrame('BUY', index = prices.index, columns = ['Order'])
    symbol_df = pd.DataFrame(symbol, index = prices.index, columns = ['Symbol'])
    final_values = 0
    for i, row in prices.iterrows():
        buy = 0
        current_indx = prices.index.get_loc(i)
        current_price = row[symbol]
        if(current_indx < prices.shape[0] - 1):
            next_index = prices.index[current_indx + 1]
            future_price = prices.loc[next_index][symbol]
            if (future_price > current_price) and (final_values < 1000):
                buy_sell.loc[i]['Order'] = 'BUY'
                if final_values != 0:
                    orders.loc[i]['Shares'] = 2000
                    final_values += 2000
                else:
                    orders.loc[i]['Shares'] = 1000
                    final_values += 1000
            elif (future_price < current_price) and (final_values > -1000):
                buy_sell.loc[i]['Order'] = 'SELL'
                if final_values != 0:
                    orders.loc[i]['Shares'] = 2000
                    final_values = final_values - 2000
                else:
                    orders.loc[i]['Shares'] = 1000
                    final_values = final_values - 1000
    df_final = pd.concat([symbol_df, buy_sell, orders], axis=1)
    df_final.columns = ['Symbol', 'Order', 'Shares']
    df_final = df_final[ df_final.Shares != 0]
    return df_final



def fill_miss_vals( df):
    df_fill = df.fillna(method ='ffill')
    df_fill = df.fillna(method = 'bfill')

def normalize_df(prices):
    fill_miss_vals(prices)
    return prices / prices.ix[0, :]

def author():
    return 'zwin3'


def test_policy():
    start_date=dt.datetime(2008,1,1)
    end_date=dt.datetime(2009,12,31)
    sv = 100000
    df_final = testPolicy(symbol = "JPM", start_date=start_date, end_date=end_date, sv = sv)
    portvals = msim.compute_portvals(df_final, start_val = sv)

    syms = ['SPY']
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(syms, dates)
    prices_SPY = prices_all['SPY']
    prices_SPY= normalize_df(prices_SPY)
    prices_portval = normalize_df(portvals)

    df_plt = pd.concat([prices_portval, prices_SPY], axis=1)
    df_plt.columns = ['Portfolio', 'Benchmark']
    df_plt.plot(title=' Best Strategy',  color=['Black', 'Blue'])
    axis_label = df_plt.plot(title=" Best Strategy joint" )
    axis_label.set_xlabel(" Dates "), axis_label.set_ylabel(" Price ")

    #df_plt.set_xlabel(" Dates "), df_plt.set_ylabel(" Price ")
    plt.show()
    prices_SPX = get_data(['SPY'] , pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['SPY']]
    portvals_SPX = (prices_SPX/prices_SPX.ix[0,:]).sum(axis=1)

    daily_returns = portvals[1:].values / portvals[:-1] - 1
    cum_ret = portvals[-1] / portvals[0] - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(252) * avg_daily_ret / std_daily_ret  #sf = 245

    daily_returns_SPY = portvals_SPX[1:].values / portvals_SPX[:-1] - 1
    cum_ret_SPY = portvals_SPX[-1] / portvals_SPX[0] - 1
    avg_daily_ret_SPY = daily_returns_SPY.mean()
    std_daily_ret_SPY = daily_returns_SPY.std()
    sharpe_ratio_SPY = np.sqrt(252) * avg_daily_ret_SPY / std_daily_ret_SPY

    print('In Sample stats:')

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

    print('Out of Sample Stats:')


    start_date=dt.datetime(2010,1,1)
    end_date=dt.datetime(2011,12,31)
    sv = 100000
    df_final = testPolicy(symbol = "JPM", start_date=start_date, end_date=end_date, sv = sv)
    portvals = msim.compute_portvals(df_final, start_val = sv)

    syms = ['SPY']
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(syms, dates)
    prices_SPY = prices_all['SPY']
    prices_SPY = normalize_df(prices_SPY)
    prices_portval_normalized = normalize_df(portvals)

    prices_SPX = get_data(['SPY'] , pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['SPY']]
    portvals_SPX = (prices_SPX/prices_SPX.ix[0,:]).sum(axis=1)

    daily_returns = portvals[1:].values / portvals[:-1] - 1
    cum_ret = portvals[-1] / portvals[0] - 1
    avg_daily_ret = daily_returns.mean()
    std_daily_ret = daily_returns.std()
    sharpe_ratio = np.sqrt(252) * avg_daily_ret / std_daily_ret  #sf = 245

    daily_returns_SPY = portvals_SPX[1:].values / portvals_SPX[:-1] - 1
    cum_ret_SPY = portvals_SPX[-1] / portvals_SPX[0] - 1
    avg_daily_ret_SPY = daily_returns_SPY.mean()
    std_daily_ret_SPY = daily_returns_SPY.std()
    sharpe_ratio_SPY = np.sqrt(252) * avg_daily_ret_SPY / std_daily_ret_SPY

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
    test_policy()
