import numpy as np
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
from util import get_data, plot_data
import matplotlib.pyplot as plt
# def indicator(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
def indicator(symbol, sd, ed):
    """
    parameters:
    symbol: the stock symbol to act on
    sd: A DateTime object that represents the start date
    ed: A DateTime object that represents the end date
    sv: Start value of the portfolio
    Return: None
    """
    sd_m_1m = sd + dt.timedelta(days = -300)
    ed_p_1m = ed + dt.timedelta(days = 30)
    # print(sd_m_1m)
    # import price df - adding 300 more days prior to start date to create moving avgs
    df = get_data([symbol], pd.date_range(sd_m_1m, ed))
    df = df[symbol].to_frame()  # remove SPY
    #BOLLINGER BAND %
    df['SMA20'] = df[symbol].rolling(20).mean() #Simple Moving Average - 20 days
    df['STD20'] = df[symbol].rolling(20).std() #Standard Deviation - 20 days
    df['Bollinger Lower Band'] = df['SMA20'] - 2 * df['STD20'] #Bollinger Lower Band 2 std
    df['Bollinger Upper Band'] = df['SMA20'] + 2 * df['STD20'] #Bollinger Upper Band 2 std
    df['Bollinger Band %'] = (df[symbol] - df['SMA20']) / (2 * df['STD20']) #Bollinger Upper Band 2 std
    conditions = [(df['Bollinger Band %'] < -.9), df['Bollinger Band %'] > .9]
    choices = [1, -1]
    # 1:long position, -1: short position
    df['Bollinger Band Signal'] = np.select(conditions, choices, default=np.nan)
    df['Bollinger Band Signal'] = df['Bollinger Band Signal'].ffill()
    #GOLDENCROSS
    df['SMA100'] = df[symbol].rolling(100).mean()  # Simple Moving Average - 100 days
    df['SMA50'] = df[symbol].rolling(50).mean()  # Simple Moving Average - 50 days
    df['Delta SMA50 - SMA100'] = df['SMA50'] -  df['SMA100']  # Delta Simple Moving Average - 50 days vs. 100 days
    df['Delta SMA50 - SMA100 shifted'] = df['Delta SMA50 - SMA100'].shift(1)  # Delta Simple Moving Average - 50 days vs. 100 days - shift 1
    # conditions = [(df['Delta SMA50 - SMA 100'] < 0) & (df['Delta SMA50 - SMA 100 shifted'] > 0), (df['Delta SMA50 - SMA 100'] > 0) & (df['Delta SMA50 - SMA 100 shifted'] < 0)]
    conditions = [(df['Delta SMA50 - SMA100'] < 0), df['Delta SMA50 - SMA100'] > 0]
    choices = [-1,1]
    # 1:long position, -1: short position
    df['Golden Cross'] = np.select(conditions, choices, np.nan)
    # df['Golden Cross'] = np.where(conditions,choices, np.nan)
    # print('df_golden_cross \n', df_golden_cross)

    #Stochastic Oscillator
    # df['H14'] = df[symbol].shift(1).rolling(14).max()
    # df['L14'] = df[symbol].shift(1).rolling(14).min()
    df['H14'] = df[symbol].rolling(14).max()
    df['L14'] = df[symbol].rolling(14).min()
    df['%K'] = (df[symbol]-df['L14'])/(df['H14']-df['L14'])
    df['%D'] = df['%K'].rolling(3).mean()
    df['Stochastic Oscillator'] = df['%D'].rolling(3).mean()
    conditions = [(df['Stochastic Oscillator'] < 0.2), df['Stochastic Oscillator'] > 0.8]
    choices = [1,-1]
    # 1:long position, -1: short position
    df['Stochastic Oscillator Signal'] = np.select(conditions, choices, default=np.nan)
    df['Stochastic Oscillator Signal'] = df['Stochastic Oscillator Signal'].ffill()
    # print('df \n', df)

    #MACD
    # df['SMA12'] = df[symbol].rolling(12).mean()
    # df['EMA12'] = 0
    # df['EMA12'].iloc[11] =  df[symbol].iloc[0:12].mean()
    # k = 2.0/(12+1) #smoothing factor
    # df['EMA12'].iloc[12:] = (df[symbol].iloc[12:] * k)\ + df['EMA12'].iloc[12:].shift(1).value * (1 - k)
    df['EMA12'] = df[symbol].ewm(span= 12 ).mean()
    df['EMA26'] = df[symbol].ewm(span= 26 ).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD Histogram'] = df['MACD'] - df['MACD Signal']
    conditions = [(df['MACD Histogram'] < 0.), df['MACD Histogram'] > 0.]
    choices = [-1,1]
    # 1:long position, -1: short position
    df['MACD Learner Signal'] = np.select(conditions, choices, default=np.nan)
    df['MACD Learner Signal'] = df['MACD Learner Signal'].ffill()

    #RSI
    df['Price Change'] = df[symbol] - df[symbol].shift(1)
    df['Gain'] = np.where(df['Price Change'] >=0, df['Price Change'],0)
    df['SMA14 Gain'] = df['Gain'].rolling(14).mean()
    df['Loss'] = np.where(df['Price Change'] <0, -df['Price Change'],0)
    df['SMA14 Loss'] = df['Loss'].rolling(14).mean()
    df['RSI'] = np.where(df['SMA14 Loss'] != 0, 1-1/(1+df['SMA14 Gain']/df['SMA14 Loss']),1)
    conditions = [(df['RSI'] < 0.3), df['RSI'] > 0.7]
    choices = [1,-1]
    # 1:long position, -1: short position
    df['RSI Signal'] = np.select(conditions, choices, default=np.nan)
    # df['RSI Signal'] = df['RSI Signal'].ffill()
    # # print('df \n', df)

    df.drop(df[df.index < sd].index, inplace = True)
    ######################################################################################################
    #-----------------------------------Plot bollinger bands---------------------------------------------#
    ######################################################################################################
    plt.subplot(211)
    plt.title(symbol + ' Bollinger Band - 20 days periods 2 Stdev',size = 8)
    plt.ylabel('Stock Price $', size = 8)
    # plt.xlabel('Date', size=8)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.plot(df.index.tolist(), df[symbol] , label=symbol, linewidth=.8)
    plt.plot(df.index.tolist(), df['SMA20'] , label='SMA20', linewidth=.8 )
    plt.plot(df.index.tolist(), df['Bollinger Lower Band'] , label='Bollinger Lower Band', linewidth=.8 )
    plt.plot(df.index.tolist(), df['Bollinger Upper Band'] , label='Bollinger Upper Band', linewidth=.8 )
    plt.xticks(size = 7 )
    plt.yticks(size = 7 )
    plt.gca().set_xlim(sd, ed)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.legend(prop={'size': 7})
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0)
    #Plot BB%
    plt.subplot(212)
    # plt.title(symbol + ' Bollinger Band %',size=9)
    plt.ylabel('BB%', size = 8)
    plt.xlabel('Date', size = 8)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.plot(df.index.tolist(), df['Bollinger Band %'] , label='Bollinger Band %', linewidth=.8, color = 'purple' )
    plt.legend(prop={'size': 7})
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.gca().set_xlim(sd, ed)
    # plt.savefig('images/Bollinger_Bands.png')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0)
    # plt.show()
    plt.clf()
    ######################################################################################################
    #-----------------------------------Plot Golden Cross------------------------------------------------#
    ######################################################################################################
    plt.subplot(211)
    plt.title(symbol + ' Golden Cross',size = 8)
    plt.ylabel('Stock Price $', size = 8)
    # plt.xlabel('Date', size = 7)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.plot(df.index.tolist(), df[symbol] , label=symbol, linewidth=.8)
    plt.plot(df.index.tolist(), df['SMA100'] , label='SMA100', linewidth=.8 )
    plt.plot(df.index.tolist(), df['SMA50'] , label='SMA50', linewidth=.8 )
    plt.xticks(size = 7 )
    plt.yticks(size=7)
    plt.gca().set_xlim(sd, ed)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.legend(prop={'size': 7})

    #Plot Delta SMAs
    plt.subplot(212)
    # plt.title(symbol + ' Bollinger Band %',size=9)
    plt.ylabel('Delta SMA50 - SMA100', size=8)
    plt.xlabel('Date', size=8)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.plot(df.index.tolist(), df['Delta SMA50 - SMA100'] , label='Delta SMA50 - SMA 100', linewidth=.8, color='purple' )
    plt.fill_between(df.index.tolist() , 0, df['Delta SMA50 - SMA100'], where=df['Delta SMA50 - SMA100'] <0, color='red')
    plt.fill_between(df.index.tolist() , 0, df['Delta SMA50 - SMA100'], where=df['Delta SMA50 - SMA100'] >0, color='green')
    plt.legend(prop={'size': 7})
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.gca().set_xlim(sd, ed)
    # plt.savefig('images/Golden Cross.png')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0)
    # plt.show()
    plt.clf()
    ######################################################################################################
    # -----------------------------------Stochastic Oscillator-------------------------------------------#
    ######################################################################################################
    plt.subplot(211)
    plt.title(symbol + ' Stochastic Oscillator', size=8)
    plt.ylabel('Stock Price $', size=8)
    # plt.xlabel('Date', , size=8)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.plot(df.index.tolist(), df[symbol], label=symbol, linewidth=.8)
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.gca().set_xlim(sd, ed)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.legend(prop={'size': 7})

    # Plot Delta SMAs
    plt.subplot(212)
    plt.ylabel('Stochastic Oscillator',  size=8)
    plt.xlabel('Date', size=8)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.plot(df.index.tolist(), df['Stochastic Oscillator'], label='Stochastic Oscillator', linewidth=.8, color='purple')
    plt.plot(df.index.tolist(),np.full((df.shape[0],1),0.8),color='grey', linewidth=.8)
    plt.plot(df.index.tolist(),np.full((df.shape[0],1),0.2),color='grey', linewidth=.8)
    plt.gca().axes.set_yticks([0, 0.2, 0.8, 1])
    # plt.fill_between(df.index.tolist(), 0.8,  df['Stochastic Oscillator'],  where=df['Stochastic Oscillator'] >=.8, color='green')
    # plt.fill_between(df.index.tolist(), 0.2,  df['Stochastic Oscillator'],  where=df['Stochastic Oscillator'] <=.2, color='red')
    # plt.fill_between(df.index.tolist() , 0, df['Stochastic Oscillator'], where=df['Stochastic Oscillator'] <.2, color='red')
    # plt.fill_between(df.index.tolist() , 0, df['Stochastic Oscillator'], where=df['Stochastic Oscillator'] >.8, color='green')

    plt.legend(prop={'size': 7})
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.gca().set_xlim(sd, ed)
    # plt.savefig('images/Stochastic Oscillator.png')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0)
    # plt.show()
    plt.clf()
    ######################################################################################################
    # -----------------------------------MACD------------------------------------------------------------#
    ######################################################################################################
    plt.subplot(211)
    plt.title(symbol + ' MACD', size=8)
    plt.ylabel('Stock Price $', size=8)
    # plt.xlabel('Date', , size=8)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.plot(df.index.tolist(), df[symbol], label=symbol, linewidth=.8)
    plt.plot(df.index.tolist(), df['EMA12'], label='EMA12', linewidth=.8)
    plt.plot(df.index.tolist(), df['EMA26'], label='EMA26', linewidth=.8)
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.gca().set_xlim(sd, ed)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.legend(prop={'size': 7})

    # Plot MACD
    plt.subplot(212)
    plt.ylabel('MACD',  size=8)
    plt.xlabel('Date', size=8)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.plot(df.index.tolist(), df['MACD'], label='MACD', linewidth=.8)
    plt.plot(df.index.tolist(), df['MACD Signal'], label='MACD Signal', linewidth=.8)
    plt.bar(df.index.tolist(), df['MACD Histogram'] , label='MACD Histogram', linewidth=1, color='purple')

    # plt.fill_between(df.index.tolist(), 0.8,  df['Stochastic Oscillator'],  where=df['Stochastic Oscillator'] >=.8, color='green')
    # plt.fill_between(df.index.tolist(), 0.2,  df['Stochastic Oscillator'],  where=df['Stochastic Oscillator'] <=.2, color='red')
    # plt.fill_between(df.index.tolist() , 0, df['Stochastic Oscillator'], where=df['Stochastic Oscillator'] <.2, color='red')
    # plt.fill_between(df.index.tolist() , 0, df['Stochastic Oscillator'], where=df['Stochastic Oscillator'] >.8, color='green')

    plt.legend(prop={'size': 7})
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.gca().set_xlim(sd, ed)
    # plt.savefig('images/MACD.png')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0)

    # plt.show()
    plt.clf()
    ######################################################################################################
    # -----------------------------------RSI Relative Strength Index-------------------------------------#
    ######################################################################################################
    plt.subplot(211)
    plt.title(symbol + ' Relative Strength Index - RSI', size=8)
    plt.ylabel('Stock Price $', size=8)
    # plt.xlabel('Date', , size=8)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.plot(df.index.tolist(), df[symbol], label=symbol, linewidth=.8)
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.gca().set_xlim(sd, ed)
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.legend(prop={'size': 7})

    # Plot RSI
    plt.subplot(212)
    plt.ylabel('RSI',  size=8)
    plt.xlabel('Date', size=8)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.plot(df.index.tolist(), df['RSI'], label='RSI', linewidth=.8, color='purple')
    plt.plot(df.index.tolist(),np.full((df.shape[0],1),0.7),color='grey', linewidth=.8)
    plt.plot(df.index.tolist(),np.full((df.shape[0],1),0.3),color='grey', linewidth=.8)
    plt.gca().axes.set_yticks([0, 0.3, 0.7, 1])
    # plt.fill_between(df.index.tolist(), 0.8,  df['Stochastic Oscillator'],  where=df['Stochastic Oscillator'] >=.8, color='green')
    # plt.fill_between(df.index.tolist(), 0.2,  df['Stochastic Oscillator'],  where=df['Stochastic Oscillator'] <=.2, color='red')
    # plt.fill_between(df.index.tolist() , 0, df['Stochastic Oscillator'], where=df['Stochastic Oscillator'] <.2, color='red')
    # plt.fill_between(df.index.tolist() , 0, df['Stochastic Oscillator'], where=df['Stochastic Oscillator'] >.8, color='green')

    plt.legend(prop={'size': 7})
    plt.xticks(size=7)
    plt.yticks(size=7)
    plt.gca().set_xlim(sd, ed)
    # plt.savefig('images/RSI.png')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0)
    # plt.show()
    plt.clf()
    return df
    # return df[['Bollinger Band Signal','Stochastic Oscillator Signal','MACD Learner Signal','Golden Cross','RSI Signal']]
def author():
  return ''
if __name__ == "__main__":
    df = indicator(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31))
    # print(df)
