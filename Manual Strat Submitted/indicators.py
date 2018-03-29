import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
from util import get_data, plot_data

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

def test():
    start_date = '2008-1-1'
    end_date = '2009-12-31'
    dates = pd.date_range(start_date,end_date)
    symbols = 'JPM'
    window_size = 20
    df = get_data([symbols],pd.date_range(start_date,end_date),False)
    df = fill_miss_vals( df )
    df = df / df.ix[0,]
    sma =   df[symbols].rolling(window_size).mean()
    std_val = df[symbols].rolling(window_size).std()
    upper_band, lower_band = bollinger_bands(sma,std_val)
    sma = df[symbols].rolling(window_size).mean()
    bb = bb_value(df[symbols],window_size)
    Momentum = momentum_value(df[symbols],window_size)
    indx_div = df.divide(sma, axis='index')
    symbol = symbols[0]
    #price_sma = pd.DataFrame(0, index = df.index, columns = ['Price/SMA'])
    #price_sma['Price/SMA'] = df[symbol]/sma['SMA']
    price_sma = df/sma
    #sma_plot = pd.concat([df, sma, price_sma], axis=1)
    #plot_data(pd.concat([df, sma, price_sma], axis=1), title="Simple Moving Average", xlabel="Date", ylabel="Stock Price")
    #plt.show()
    #sma_plot.columns = [symbols, 'SMA', 'Price/SMA']
    #sma_plot.plot(grid=True, title='Simple Moving Average', use_index=True)
    #plt.show()




    sma_plot = plt.figure(1)
    joint = df.join(sma, lsuffix = ' Price', rsuffix = 'SMA').join(indx_div, lsuffix = '', rsuffix = 'Price/sma')
    joint.columns = [' Price','sma','Price/sma']
    ax = joint.plot(title=" Simple Moving Average" )
    ax.set_xlabel(" Dates "), ax.set_ylabel(" Price ")
    sma_plot.show()
    #sma_plot = pd.concat([df, sma, price_sma], axis=1)
    #sma_plot.columns = [symbol, 'SMA', 'Price SMA ratio']
    #sma_plot.plot( title='Simple Moving Average')
    #plot_data()
    bb_plot = plt.figure(2)
    joint = df.join(sma, lsuffix = ' Price', rsuffix = 'SMA').join(upper_band, lsuffix = '_', rsuffix = 'upperband').join(lower_band, lsuffix = '_', rsuffix = 'lowerband')
    joint.columns = [' Price','sma','Upper Bands', 'Lower Bands']
    ax = joint.plot(title="Bollinger bands")
    ax.set_xlabel(" Dates "), ax.set_ylabel(" Price ")
    bb_plot.show()
    #bb_plot = pd.concat([df, sma, price_sma], axis=1)
    #bb_plot.columns = [symbol, 'SMA', 'Price_SMA_ratio']
    #bb_plot.plot( title='Simple Moving Average')
    #plot_data()
    momentum_plot = plt.figure(3)
    joint = df.join(Momentum, lsuffix = ' Price', rsuffix = 'Momentum')
    joint.columns = [' Price', 'Momentum']
    ax = joint.plot(title="Momentum")
    ax.set_xlabel(" Dates ") ,ax.set_ylabel(" Price ")
    momentum_plot.show()

    plt.show()

if __name__ == "__main__":
    test()
