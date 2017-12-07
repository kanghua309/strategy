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
end   = '2017-9-11'  # 必须在国内交易日

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
    returns = Returns(inputs=[USEquityPricing.close], window_length=5, mask = private_universe)  # 预测一周数据
    ######################################################################################################
    ep = 1/Fundamental(mask = private_universe,asset_finder=asset_finder).pe.latest
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
        'vol10': vol10.zscore(groupby=sector).downsample('week_start'),
        'rev10': rev10.zscore(groupby=sector).downsample('week_start'),
        'vol20': vol20.zscore(groupby=sector).downsample('week_start'),
        'rev20': rev20.zscore(groupby=sector).downsample('week_start'),
        'vol30':vol30.zscore(groupby=sector).downsample('week_start'),
        'rev30':rev30.zscore(groupby=sector).downsample('week_start'),

        'ILLIQ5':illiq5.zscore(groupby=sector,mask=illiq5.percentile_between(1, 99)).downsample('week_start'),
        'ILLIQ22':illiq22.zscore(groupby=sector, mask=illiq22.percentile_between(1, 99)).downsample('week_start'),

        'mom5'  :mom5.zscore(groupby=sector,mask=mom5.percentile_between(1, 99)).downsample('week_start'),
        'mom22': mom22.zscore(groupby=sector, mask=mom22.percentile_between(1, 99)).downsample('week_start'),

        'rsi5'  :rsi5.zscore(groupby=sector,mask=rsi5.percentile_between(1, 99)).downsample('week_start'),
        'rsi22': rsi22.zscore(groupby=sector, mask=rsi22.percentile_between(1, 99)).downsample('week_start'),

        #'sector':sector,
        #'returns':returns.quantiles(100),
        'returns': returns.downsample('week_start') * 100,
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

print result.head(10)
#print type(result)
#print result.reset_index()
#result.replace([np.inf,-np.inf],np.nan)
result = result.reset_index().drop(['level_0','level_1'],axis = 1).replace([np.inf,-np.inf],np.nan).fillna(0)
print "#####################################"
print result.head(10)
# print result.isnull().any()
# print result[result.isnull().values==True]


from sklearn.preprocessing import PolynomialFeatures
print type(result.values),result.values
print "============================="
# print np.isfinite(result.values.all())
# all_inf_or_nan = result.isin([np.inf,-np.inf,np.nan]).all(axis = 'columns')
# x = result[~all_inf_or_nan]
# # from sklearn.preprocessing import Imputer
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imp.fit(X)
#
# quadratic_featurizer  = PolynomialFeatures(interaction_only=True)
# X_train_quadratic = quadratic_featurizer.fit_transform(result)
# print X_train_quadratic
# print np.shape(X_train_quadratic),type(X_train_quadratic)
#
# print quadratic_featurizer.get_feature_names(result.columns),len(quadratic_featurizer.get_feature_names(result.columns))





X = result.drop('returns', 1)
Y = result['returns']

print ("total data size :",len(result))
test_size=2000
Train_X = X[:-test_size].values
Train_Y = Y[:-test_size].values
Test_X  = X[-test_size:].values
Test_Y  = Y[-test_size:].values

print ("*******************************************")
# print Test_Y.head(10),len(Test_Y)
# print Test_X.head(10),len(Test_X)
# #
# #
# # TRAIN_X = X[0:]
# # TRAIN_Y = array[0:400,13]
# #
# # TEST_X = array[400:,0:13]
# # TEST_Y = array[400:,13]
#
#
# print X.head(10)
# print Y.head(10)

#quadratic_featurizer  = PolynomialFeatures(interaction_only=True)
#Train_X = quadratic_featurizer.fit_transform(Train_X)
#Test_X = quadratic_featurizer.fit_transform(Test_X)

# Train_Y = np.diff(Train_Y)
# Test_Y = np.diff(Test_Y)
# import scipy.stats as stats
# Train_Y = stats.boxcox(Train_Y)[0]
# Test_Y = stats.boxcox(Test_Y)[0]

#print X_train_quadratic
#print np.shape(X_train_quadratic),type(X_train_quadratic)
# print quadratic_featurizer.get_feature_names(result.columns),len(quadratic_featurizer.get_feature_names(result.columns))


from modeltest import model_cross_valid,model_fit_and_test
model_cross_valid(Train_X[0:],Train_Y)
print ("---------------fit and test-------------")
model_fit_and_test(Train_X[0:],Train_Y,Test_X[0:],Test_Y)





#print type(result.as_matrix()),result.as_matrix
print ("-------------------------------------------------------------------------")
