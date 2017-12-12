# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:42:21 2017

@author: kanghua
"""
import numpy as np
import pandas as pd

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor, Returns, Latest ,RSI
from me.pipeline.factors.tsfactor import Fundamental
from me.pipeline.classifiers.tushare.sector import get_sector,RandomUniverse,get_sector_class,get_sector_by_onehot
from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.classifiers import CustomClassifier,Latest
from me.pipeline.factors.boost import Momentum
import tushare as ts

from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Returns,AverageDollarVolume

from me.helper.research_env import Research
from me.pipeline.factors.tsfactor import Fundamental
from me.pipeline.filters.universe import make_china_equity_universe, default_china_equity_universe_mask, \
    private_universe_mask
from zipline.utils.cli import Date, Timestamp

start = '2016-8-10'   # 必须在国内交易日
end   = '2017-8-11'  # 必须在国内交易日

c,_ = get_sector_class()
ONEHOTCLASS = tuple(c)

hs300 = ts.get_hs300s()['code']
#print type(hs300),hs300
#print hs300.tolist()


class ILLIQ(CustomFactor):
    inputs = [USEquityPricing.close,USEquityPricing.volume]
    window_length = int(252)
    def compute(self, today, assets, out, close,volume):
        window_length = len(close)
        _rets = np.abs(pd.DataFrame(close, columns=assets).pct_change()[1:])
        _vols  = pd.DataFrame(volume, columns= assets)[1:]
        out[:] =(_rets/_vols).mean().values

def make_pipeline(asset_finder):

    private_universe = private_universe_mask( hs300.tolist(),asset_finder=asset_finder)
    #print private_universe_mask(['000001','000002','000005'],asset_finder=asset_finder)
    ######################################################################################################
    returns = Returns(inputs=[USEquityPricing.close], window_length=5)  # 预测一周数据
    ######################################################################################################
    ep = 1/Fundamental(mask = private_universe,asset_finder=asset_finder).pe
    bp = 1/Fundamental(mask = private_universe,asset_finder=asset_finder).pb
    bvps = Fundamental(mask = private_universe,asset_finder=asset_finder).bvps
    market = Fundamental(mask = private_universe,asset_finder=asset_finder).outstanding
    totals = Fundamental(mask = private_universe,asset_finder=asset_finder).totals
    totalAssets = Fundamental(mask = private_universe,asset_finder=asset_finder).totalAssets
    fixedAssets = Fundamental(mask = private_universe,asset_finder=asset_finder).fixedAssets
    esp = Fundamental(mask = private_universe,asset_finder=asset_finder).esp
    rev = Fundamental(mask = private_universe,asset_finder=asset_finder).rev
    profit = Fundamental(mask = private_universe,asset_finder=asset_finder).profit
    gpr = Fundamental(mask = private_universe,asset_finder=asset_finder).gpr
    npr = Fundamental(mask = private_universe,asset_finder=asset_finder).npr

    rev10 = Returns(inputs=[USEquityPricing.close], window_length=10,mask = private_universe)
    vol10 = AverageDollarVolume(window_length=20,mask = private_universe)
    rev20 = Returns(inputs=[USEquityPricing.close], window_length=20,mask = private_universe)
    vol20 = AverageDollarVolume(window_length=20,mask = private_universe)
    rev30 = Returns(inputs=[USEquityPricing.close], window_length=30,mask = private_universe)
    vol30 = AverageDollarVolume(window_length=20,mask = private_universe)

    illiq22 = ILLIQ(window_length=22,mask = private_universe)
    illiq5 = ILLIQ(window_length=5,mask = private_universe)

    rsi5 = RSI(window_length=5,mask = private_universe)
    rsi22 = RSI(window_length=22,mask = private_universe)

    mom5 = Momentum(window_length=5,mask = private_universe)
    mom22 = Momentum(window_length=22,mask = private_universe)


    sector = get_sector(asset_finder=asset_finder,mask=private_universe)
    ONEHOTCLASS,sector_indict_keys = get_sector_by_onehot(asset_finder=asset_finder,mask=private_universe)

    pipe_columns = {

        'ep':ep.zscore(groupby=sector).downsample('month_start'),
        'bp':bp.zscore(groupby=sector).downsample('month_start'),
        'bvps':bvps.zscore(groupby=sector).downsample('month_start'),
        'market_cap': market.zscore(groupby=sector).downsample('month_start'),
        'totals': totals.zscore(groupby=sector).downsample('month_start'),
        'totalAssets': totalAssets.zscore(groupby=sector).downsample('month_start'),
        'fixedAssets': fixedAssets.zscore(groupby=sector).downsample('month_start'),
        'esp': esp.zscore(groupby=sector).downsample('month_start'),
        'rev': rev.zscore(groupby=sector).downsample('month_start'),
        'profit': profit.zscore(groupby=sector).downsample('month_start'),
        'gpr': gpr.zscore(groupby=sector).downsample('month_start'),
        'npr': npr.zscore(groupby=sector).downsample('month_start'),
        'vol10': vol10.zscore(groupby=sector),
        'rev10': rev10.zscore(groupby=sector),
        'vol20': vol20.zscore(groupby=sector),
        'rev20': rev20.zscore(groupby=sector),
        'vol30':vol30.zscore(groupby=sector),
        'rev30':rev30.zscore(groupby=sector),

        'ILLIQ5':illiq5.zscore(groupby=sector,mask=illiq5.percentile_between(1, 99)),
        'ILLIQ22':illiq22.zscore(groupby=sector, mask=illiq22.percentile_between(1, 99)),

        'mom5'  :mom5.zscore(groupby=sector,mask=mom5.percentile_between(1, 99)),
        'mom22': mom22.zscore(groupby=sector, mask=mom22.percentile_between(1, 99)),

        'rsi5'  :rsi5.zscore(groupby=sector,mask=rsi5.percentile_between(1, 99)),
        'rsi22': rsi22.zscore(groupby=sector, mask=rsi22.percentile_between(1, 99)),

        #'sector':sector,
        #'returns':returns.quantiles(100),
        'returns': returns * 100,
    }
    # pipe_screen = (low_returns | high_returns)
    pipe = Pipeline(columns=pipe_columns,
           screen=private_universe,
           )
    i = 0
    for c in ONEHOTCLASS:
        pipe.add(c,sector_indict_keys[i])
        i +=1
    return pipe


pd.set_option('display.width', 8000)
research = Research()
#print(research.get_engine()._finder)
my_pipe = make_pipeline(research.get_engine()._finder)
result = research.run_pipeline(my_pipe,
                               Date(tz='utc', as_timestamp=True).parser(start),
                               Date(tz='utc', as_timestamp=True).parser(end))

result = result.reset_index().drop(['level_0','level_1'],axis = 1).replace([np.inf,-np.inf],np.nan).fillna(0)
print "############################################"


X = result.drop('returns', 1)
Y = result['returns']

print ("total data size :",len(result))
#test_size=2000
Train_X = X.values
Train_Y = Y.values
# Test_X  = X[-test_size:].values
# Test_Y  = Y[-test_size:].values

print ("*******************************************")
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
model = LinearRegression()
model.fit(Train_X, Train_Y)
print ("******************************************* Train Over .......")


start = '2017-8-14'   # 必须在国内交易日
end   = '2017-9-29'   # 必须在国内交易日
result = research.run_pipeline(my_pipe,
                               Date(tz='utc', as_timestamp=True).parser(start),
                               Date(tz='utc', as_timestamp=True).parser(end))

result = result.reset_index().drop(['level_0','level_1'],axis = 1).replace([np.inf,-np.inf],np.nan).fillna(0)
X = result.drop('returns', 1)
Y = result['returns']
predict = model.predict(X.values)

data = result.join(pd.DataFrame({"Predict Factor":predict}))
print data.head(10)

ranked_data = data.sort('Predict Factor')

print ("******************************************* Predict Over .......")

number_of_baskets = 300/30
basket_returns = np.zeros(number_of_baskets)

for i in range(number_of_baskets):
    start = i * 30
    end = i * 30 + 30
    basket_returns[i] = ranked_data[start:end]['returns'].mean()

basket_returns[number_of_baskets-1] - basket_returns[0]

import matplotlib.pyplot as plt
plt.bar(range(number_of_baskets), basket_returns)
plt.ylabel('Returns')
plt.xlabel('Basket')
plt.legend(['Returns of Each Basket']);
plt.show()
# Plot the returns of each basket
# plt.bar(range(number_of_baskets), basket_returns)
# plt.ylabel('Returns')
# plt.xlabel('Basket')
# plt.legend(['Returns of Each Basket']);


