from util import get_data, plot_data
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode as msim
plt.switch_backend('agg')
def author():
    return 'zwin3'

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


def testPolicy(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
    syms = [symbol]
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices_SPY = prices_all['SPY']
    prices_SPY = normalize_df(prices_SPY)
    prices = prices_all[syms]
    prices = normalize_df(prices)
    window_sizes = 20
    momentum = pd.DataFrame(0, index = prices.index, columns = ['Momentum'])
    momentum['Momentum'] = prices.diff(window_sizes)/prices.shift(window_sizes)
    sma = pd.DataFrame(0, index = prices.index, columns = ['SMA'])
    sma['SMA'] = prices.rolling(window=window_sizes, min_periods = window_sizes).mean()   #sma value
    prices_sma_ratio = pd.DataFrame(0, index = sma.index, columns = ['Price_SMA_ratio'])
    prices_sma_ratio['Price_SMA_ratio'] = prices[symbol]/sma['SMA']
    bb = pd.DataFrame(0, index = prices.index, columns =  ['lower', 'upper'])
    bands = pd.DataFrame(0, index = prices.index, columns = ['band'])
    bands['band'] = prices.rolling(window = window_sizes, min_periods = window_sizes).std()
    bb['upper'] = sma['SMA'] + (bands['band'] * 2)
    bb['lower'] = sma['SMA'] - (bands['band'] * 2)
    bb_percentage = pd.DataFrame(0, index = prices.index, columns = ['BBP'])
    bb_percentage['BBP'] = (prices[symbol] - bb['lower']) / (bb['upper'] - bb['lower'])
    # made the values of  the three indicators


    orders = pd.DataFrame(0, index = prices.index, columns = ['Shares'])
    buy_sell = pd.DataFrame('BUY', index = prices.index, columns = ['Order'])
    symbol_df = pd.DataFrame(symbol, index = prices.index, columns = ['Symbol'])
    final_values = 0
    for i, row in prices.iterrows():
        sma_value = sma.loc[i]['SMA']
        sma_ratio = prices_sma_ratio.loc[i]['Price_SMA_ratio']
        bbp_value = bb_percentage.loc[i]['BBP']
        momentum_value = momentum.loc[i]['Momentum']
        current_price = row[symbol]
        if (momentum_value < -0.05) and (bbp_value < 0.15) and (final_values < 1000):
            buy_sell.loc[i]['Order'] = 'BUY'
            if final_values != 0:
                orders.loc[i]['Shares'] = 2000
                final_values += 2000
            else:
                orders.loc[i]['Shares'] = 1000
                final_values += 1000
        elif (momentum_value > 0.05) and (bbp_value > 0.85) and (final_values > -1000):
            buy_sell.loc[i]['Order'] = 'SELL'
            if final_values != 0:
                orders.loc[i]['Shares'] = 2000
                final_values = final_values - 2000
            else:
                orders.loc[i]['Shares'] = 1000
                final_values = final_values - 1000
    df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
    df_trades.columns = ['Symbol', 'Order', 'Shares']
    df_trades = df_trades[df_trades.Shares != 0]
    return df_trades

def test_code():
    start_date=dt.datetime(2008,1,1)
    end_date=dt.datetime(2009,12,31)
    sv = 100000
    df_trades = testPolicy(symbol = "JPM", sd = start_date, ed = end_date, sv = sv)
    portvals = msim.compute_portvals(df_trades, start_val = sv)
    syms = ['SPY']
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(syms, dates)
    prices_SPY = prices_all['SPY']
    prices_SPY = normalize_df(prices_SPY)
    prices_portval_normalized = normalize_df(portvals)

    chart_df = pd.concat([prices_portval_normalized, prices_SPY], axis=1)
    chart_df.columns = ['Portfolio', 'Benchmark']
    chart_df.plot( title=' Insample: Manual strategy ', use_index=True, color=['Black', 'Blue'])
    for index, row in df_trades.iterrows():
        if df_trades.loc[index]['Order'] == 'BUY':
            plt.axvline(x=index, color='g', linestyle='-')
        elif df_trades.loc[index]['Order'] == 'SELL':
            plt.axvline(x=index, color='r', linestyle='-')

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
    sharpe_ratio_SPY = np.sqrt(252) * avg_daily_ret_SPY / std_daily_ret_SPY  #sf = 252
    print('In Sample :')
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
    print('Out of Sample :')


    start_date=dt.datetime(2010,1,1)
    end_date=dt.datetime(2011,12,31)
    sv = 100000
    df_trades = testPolicy(symbol = "JPM", sd = start_date, ed = end_date, sv = sv)
    portvals = msim.compute_portvals( df_trades, start_val = sv)
    syms = ['SPY']
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(syms, dates)
    prices_SPY = prices_all['SPY']
    prices_SPY = normalize_df(prices_SPY)
    prices_portval_normalized = normalize_df(portvals)
    chart_df = pd.concat([prices_portval_normalized, prices_SPY], axis=1)
    chart_df.columns = ['Portfolio', 'Benchmark']
    chart_df.plot( title=' Outsample: Manual strategy ', use_index=True, color=['Black', 'Blue'])
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
    plt.show()

if __name__ == "__main__":
    test_code()
