import StrategyLearner as sl
import numpy as np
import datetime as dt
import ManualStrategy as ms
import pandas as pd
from util import get_data
import matplotlib.pyplot as plt
import marketsimcode as mktsim
def exp3_insample(symbol, sv, commission,impact):

    # In sample comparison
    df = get_data([symbol], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))  # import price df
    strat_learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor
    strat_learner.add_evidence(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31))  # training phase
    df['trade_SL'] = strat_learner.testPolicy(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)  # testing phase
    df['Strategy Learner Portfolio Value'] = mktsim.compute_portvals(df['trade_SL'],     start_val=100000, commission=commission,impact=impact, )
    man_strat = ms.ManualStrategy()
    df['trade_MS'] = man_strat.testPolicy(symbol = symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)
    df['Manual Strategy Portfolio Value'] = mktsim.compute_portvals(df['trade_MS'],     start_val=100000, commission=commission,impact=impact, )

    # df['holding'] = df['trade'].cumsum()
    # df['cost'] = - df['trade'] * df['JPM']
    # df['cash'] = sv + df['cost'].cumsum()
    # df['value'] = df['cash'] + df['holding'] * df['JPM']
    df['Benchmark'] = df[symbol] / df[symbol][0]
    # df['Strategy Learner'] = df['value'] / df['value'][0]
    # print('\n', df)

    fig, ax = plt.subplots()
    ax.set_title(symbol + " In Sample - Manual vs. Strategy Learner")
    ax.set_ylabel('Normalized Portfolio Value',size = 8)
    ax.set_xlabel('Date',size = 8)
    # ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.plot(df.index.tolist(), df['Benchmark'] , label='Benchmark', color='purple', linewidth=.8 )
    ax.plot(df.index.tolist(), df['Strategy Learner Portfolio Value']/sv , label='Strategy Learner ', color='green' , linewidth=.8)
    ax.plot(df.index.tolist(), df['Manual Strategy Portfolio Value']/sv , label='Manual Strategy ', color='red' , linewidth=.8)
    plt.xticks(size = 7 )
    plt.yticks(size = 7 )
    plt.gca().set_xlim(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
    # plt.gca().axes.xaxis.set_ticklabels([])
    ax.legend()
    # plt.xticks(rotation=25)
    plt.savefig('images/' + str(symbol) + 'Experiment 1 In Sample.png')
    plt.clf()


def exp3_outsample(symbol, sv, commission,impact):

    # In sample comparison
    df = get_data([symbol], pd.date_range(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)))  # import price df
    strat_learner = sl.StrategyLearner(verbose=False, impact=impact, commission=commission)  # constructor
    strat_learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31))  # training phase
    df['trade_SL'] = strat_learner.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)  # testing phase
    df['Strategy Learner Portfolio Value'] = mktsim.compute_portvals(df['trade_SL'],     start_val=100000, commission=commission,impact=impact, )
    man_strat = ms.ManualStrategy()
    df['trade_MS'] = man_strat.testPolicy(symbol  = "JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000)
    df['Manual Strategy Portfolio Value'] = mktsim.compute_portvals(df['trade_MS'],     start_val=100000, commission=commission,impact=impact, )

    # df['holding'] = df['trade'].cumsum()
    # df['cost'] = - df['trade'] * df['JPM']
    # df['cash'] = sv + df['cost'].cumsum()
    # df['value'] = df['cash'] + df['holding'] * df['JPM']
    df['Benchmark'] = df[symbol] / df[symbol][0]
    # df['Strategy Learner'] = df['value'] / df['value'][0]
    # print('\n', df)

    fig, ax = plt.subplots()
    ax.set_title(symbol + " out Sample - Manual vs. Strategy Learner")
    ax.set_ylabel('Normalized Portfolio Value',size = 8)
    ax.set_xlabel('Date',size = 8)
    # ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.plot(df.index.tolist(), df['Benchmark'] , label='Benchmark', color='purple', linewidth=.8 )
    ax.plot(df.index.tolist(), df['Strategy Learner Portfolio Value']/sv , label='Strategy Learner ', color='green' , linewidth=.8)
    ax.plot(df.index.tolist(), df['Manual Strategy Portfolio Value']/sv , label='Manual Strategy ', color='red' , linewidth=.8)
    plt.xticks(size = 7 )
    plt.yticks(size = 7 )
    plt.gca().set_xlim(dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31))
    # plt.gca().axes.xaxis.set_ticklabels([])
    ax.legend()
    # plt.xticks(rotation=25)
    plt.savefig('images/' + str(symbol) + 'Experiment 1 out Sample.png')
    plt.clf()
def author():
  return ''

if __name__ == "__main__":
    exp3_insample(symbol = "JPM",  sv = 100000, commission = 9.95,impact = 0.005)
    exp3_outsample(symbol = "JPM",  sv = 100000, commission = 9.95,impact = 0.005)