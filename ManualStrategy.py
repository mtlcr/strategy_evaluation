import numpy as np
import indicators as indicators
import datetime as dt

class ManualStrategy(object):
    # def __init__(self, symbol, sd,  ed, sv):
    #     self.symbol = symbol
    #     self.sd = sd
    #     self.ed = ed
    #     self.sv = sv

    def testPolicy(self,symbol, sd, ed, sv):
        df = indicators.indicator(symbol = symbol, sd=sd, ed=ed)

        conditions = [(df['Bollinger Band Signal'] == 1.) & (df['Stochastic Oscillator Signal'] == 1.) & (df['RSI Signal'] == 1.),
                      (df['Bollinger Band Signal'] == 1.) & (df['Stochastic Oscillator Signal'] == 1.) & (df['MACD Learner Signal'] == 1.),
                      # (df['Bollinger Band Signal'] == 1.) & (df['Stochastic Oscillator Signal'] == 1.) & (df['Golden Cross'] == 1.),
                      (df['Bollinger Band Signal'] == -1.) & (df['Stochastic Oscillator Signal'] == -1.) & (df['RSI Signal'] == -1.),
                      (df['Bollinger Band Signal'] == -1.) & (df['Stochastic Oscillator Signal'] == -1.) & (df['MACD Learner Signal'] == -1.),
                      # (df['Bollinger Band Signal'] == -1.) & (df['Stochastic Oscillator Signal'] == -1.) & (df['Golden Cross'] == -1.)
                      ]
        choices = [1000, 1000,  -1000, -1000]

        # 1:long position, -1: short position
        df['holding'] = np.select(conditions, choices, default=np.nan)
        df['holding'] = df['holding'].ffill()
        df['holding'] = df['holding'].bfill()
        df['trade'] = df['holding'] - df['holding'].shift(1)
        df['trade'][0] = df['holding'][0]
        # print(df)
        return df['trade']

    def author(self):
        return ""

def author():
    return ""

if __name__ == "__main__":
    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)
    # print(df_trades)

