
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import datetime as dt  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import random  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np
import pandas as pd  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import util as ut
import indicators as ind
import DTLearner as DTL
import RTLearner as rl
import BagLearner as bl
class StrategyLearner(object):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        If verbose = False your code should not generate ANY output.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type verbose: bool  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type impact: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    :type commission: float  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # constructor  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Constructor method  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.verbose = verbose  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.impact = impact  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.commission = commission  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self.Learner = None
    def add_evidence(  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self,
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=10000,
    ):
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Trains your strategy learner over a given time frame.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param symbol: The stock symbol to train on  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type symbol: str  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sd: datetime  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type ed: datetime  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sv: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        # add your code to do learning here  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        # example usage of the old backward compatible util function
        # syms = [symbol]
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        # prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all["SPY"]  # only SPY, for comparison later

        df = ind.indicator(symbol=symbol, sd=sd, ed=ed)
        df['ret'] = df[symbol].shift(-1) / df[symbol] -1
        conditions = [(df['ret'] >= 0.04 + self.impact ), (df['ret'] <= -0.04 - self.impact),(df['ret'] > -0.04 - self.impact)&(df['ret'] < 0.04 + self.impact)]


        choices = [1000, -1000,0]
        # 1000:long position, -1000: short position
        df['train_y'] = np.select(conditions, choices, default=np.nan)
        # print(df)
        # train_x = df[['Bollinger Band %', 'Delta SMA50 - SMA100', 'Stochastic Oscillator', 'MACD Histogram', 'RSI']].to_numpy()
        train_x = df[['Bollinger Band %','Stochastic Oscillator', 'MACD Histogram', 'RSI']].to_numpy()
        # train_x = df[['Bollinger Band %', 'Stochastic Oscillator',  'RSI']].to_numpy()
        train_y = df['train_y'].to_numpy()
        self.LearnerBL = bl.BagLearner(rl.RTLearner, kwargs={"leaf_size": 5}, verbose=True)
        self.LearnerBL.add_evidence(train_x, train_y)
        # print(train_x)
        self.Learner = rl.RTLearner(leaf_size=5,verbose=False)  # create a RTLearner
        self.Learner.add_evidence(train_x, train_y)

    # this method should use the existing policy and test it against new data  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    def testPolicy(  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        self,  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=10000,
    ):  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        Tests your learner using data outside of the training data  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type symbol: str  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sd: datetime  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type ed: datetime  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :param sv: The starting value of the portfolio  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :type sv: int  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        :rtype: pandas.DataFrame  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        """  		  	   		  	  			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 

        df = ind.indicator(symbol=symbol, sd=sd, ed=ed)
        # index_col = df.index
        # print('index_col', index_col)
        # pred_x = df[['Bollinger Band %', 'Delta SMA50 - SMA100', 'Stochastic Oscillator', 'MACD Histogram', 'RSI']].to_numpy()
        pred_x = df[['Bollinger Band %', 'Stochastic Oscillator', 'MACD Histogram', 'RSI']].to_numpy()
        # pred_x = df[['Bollinger Band %', 'Stochastic Oscillator',  'RSI']].to_numpy()
        pred_y = self.Learner.query(pred_x)
        pred_y = pred_y.transpose()
        df_result= pd.DataFrame(pred_y,columns=['holding'])
        df_result['trade']   = df_result['holding'] - df_result['holding'].shift(1)
        df_result['trade'][0] = df_result['holding'][0]
        df_result['trade'] = df_result['trade'].fillna(0)
        df_result['id'] = df.index
        df_result.index = df_result['id']
        df_result = df_result.drop('holding', axis=1)
        df_result = df_result.drop('id', axis=1)
        # df_result.drop('id')
        # print('\n  pred_y', pred_y)
        # print('\n  df[holding]', df_result['holding'] )
        # print(type(df))
        # df_result['trade'].to_frame()
        # print(type(df_result))
        # print(' df_result\n ', df_result  )
        # print('df index \n', df )
        # print('df_result index \n', df_result.index)
        # next: convert pred_y which is holding, into trade. and return trade
        # return trades
        df_result.fillna(0)
        return df_result

    def author(self):
        return ''
def author():
    return ''

if __name__ == "__main__":
    SL = StrategyLearner( verbose=False, impact=0.005, commission=9.95)
    SL.add_evidence(symbol="JPM",sd=dt.datetime(2008, 1, 1),ed=dt.datetime(2009, 12, 31), sv = 100000)
    SL.testPolicy(symbol="JPM",sd=dt.datetime(2010, 1, 1),ed=dt.datetime(2011, 12, 31),sv=100000,)