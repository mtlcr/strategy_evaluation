import StrategyLearner as sl
import numpy as np
import datetime as dt
import ManualStrategy as ms
import pandas as pd
from util import get_data
import matplotlib.pyplot as plt
import marketsimcode as mktsim
def exp2(symbol, sv):

    # In sample comparison
    df = get_data([symbol], pd.date_range(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)))  # import price df

    # Stratergy Learner 1 - 0% impact
    strat_learner1 = sl.StrategyLearner(verbose=False, impact=0, commission=9.95)  # constructor
    strat_learner1.add_evidence(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),)  # training phase
    df['trade_SL1'] = strat_learner1.testPolicy(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                                sv=100000)  # testing phase
    df['Strategy Learner Portfolio Value 1'] = mktsim.compute_portvals(df['trade_SL1'], start_val=100000, impact=0,
                                                                       commission=9.95)

    # Stratergy Learner 2 - 1% impact
    strat_learner2 = sl.StrategyLearner(verbose=False, impact=0.005, commission=9.95)  # constructor
    strat_learner2.add_evidence(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),)  # training phase
    df['trade_SL2'] = strat_learner2.testPolicy(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                                sv=100000)  # testing phase
    df['Strategy Learner Portfolio Value 2'] = mktsim.compute_portvals(df['trade_SL2'], start_val=100000, impact=0.005,
                                                                       commission=9.95)

    # Stratergy Learner 3 - 5% impact
    strat_learner3 = sl.StrategyLearner(verbose=False, impact=0.02, commission=9.95)  # constructor
    strat_learner3.add_evidence(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),)  # training phase
    df['trade_SL3'] = strat_learner3.testPolicy(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31),
                                                sv=100000)  # testing phase
    df['Strategy Learner Portfolio Value 3'] = mktsim.compute_portvals(df['trade_SL3'], start_val=100000, impact=0.02,
                                                                       commission=9.95)

    df['Benchmark'] = df[symbol] / df[symbol][0]
    # print('\n', df)

    SL1 = df[df['trade_SL1'] != 0].count()
    # print('\n SL1 trade count', SL1)

    SL2 = df[df['trade_SL2'] != 0].count()
    # print('\n SL2 trade count', SL2)

    SL3 = df[df['trade_SL3'] != 0].count()
    # print('\n SL3 trade count', SL3)

    fig, ax = plt.subplots()
    ax.set_title("JPM In Sample - Strategy Learner behaviour with varying impact")
    ax.set_ylabel('Normalized Portfolio Value',size = 8)
    ax.set_xlabel('Date',size = 8)
    # ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    ax.plot(df.index.tolist(), df['Benchmark'] , label='Benchmark', color='purple', linewidth=.8 )
    ax.plot(df.index.tolist(), df['Strategy Learner Portfolio Value 1']/sv , label='Strategy Learner 0% impact', color='green' , linewidth=.8)
    ax.plot(df.index.tolist(), df['Strategy Learner Portfolio Value 2']/sv , label='Strategy Learner 2% impact', color='orange' , linewidth=.8)
    ax.plot(df.index.tolist(), df['Strategy Learner Portfolio Value 3']/sv , label='Strategy Learner 5% impact', color='red' , linewidth=.8)

    plt.xticks(size = 7 )
    plt.yticks(size = 7 )
    plt.gca().set_xlim(dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
    # plt.gca().axes.xaxis.set_ticklabels([])
    ax.legend()
    # plt.xticks(rotation=25)
    plt.savefig('images/Experiment 2.png')
    plt.clf()

def author():
  return ''

if __name__ == "__main__":
    exp2(symbol = "JPM",  sv = 100000)