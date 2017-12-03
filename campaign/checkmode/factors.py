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

import talib



import click
import numpy as np
import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Latest
from zipline.pipeline.factors import Returns,AverageDollarVolume
from zipline.utils.cli import Date

from me.helper.research_env import Research
from me.pipeline.factors.tsfactor import Fundamental
from me.pipeline.filters.universe import make_china_equity_universe, default_china_equity_universe_mask, \
    private_universe_mask
from zipline.utils.cli import Date, Timestamp


start = '2015-9-1'  # 必须在国内交易日
end   = '2015-9-30'  # 必须在国内交易日

c,_ = get_sector_class()
ONEHOTCLASS = tuple(c)


class ILLIQ(CustomFactor):
    inputs = [USEquityPricing.close,USEquityPricing.volume]
    window_length = int(252)
    def compute(self, today, assets, out, close,volume):
        window_length = len(close)
        _rets = np.abs(pd.DataFrame(close, columns=assets).pct_change()[1:])
        _vols  = pd.DataFrame(volume, columns= assets)[1:]
        out[:] =(_rets/_vols).mean().values

def make_pipeline(asset_finder):

    private_universe = private_universe_mask(['000001','000002','000005'],asset_finder=asset_finder)
    ######################################################################################################
    returns = Returns(inputs=[USEquityPricing.close], window_length=5)  # 预测一周数据
    ######################################################################################################
    ep = 1/Fundamental(asset_finder).pe
    bp = 1/Fundamental(asset_finder).pb
    bvps = Fundamental(asset_finder).bvps

    rev20 = Returns(inputs=[USEquityPricing.close], window_length=20,mask = private_universe)
    vol20 = AverageDollarVolume(window_length=20,mask = private_universe)
    market = Fundamental(asset_finder).outstanding

    illiq = ILLIQ(window_length=20,mask = private_universe)
    rsi = RSI(window_length=20,mask = private_universe)
    mom = Momentum()

    sector = get_sector(asset_finder=asset_finder,mask=private_universe)
    ONEHOTCLASS,sector_indict_keys = get_sector_by_onehot(asset_finder=asset_finder,mask=private_universe)


    pipe_columns = {
        # 'h2o': h2o.log1p().zscore(),
        # 'l2o': l2o.log1p().zscore(),
        # 'c2o': c2o.log1p().zscore(),
        # 'h2c': h2c.log1p().zscore(),
        # 'l2c': l2c.log1p().zscore(),
        # 'h2l': h2l.log1p().zscore(),
        # 'vol': vol.zscore(),
        # 'turnover_rate': turnover_rate.log1p().zscore(),
        # 'return': returns.log1p(),
        'ILLIQ':illiq,
        'ep':ep,
        'bp':bp,
        'bvps':bvps,
        'vol20':vol20,
        'rev20':rev20,
        'mom':mom,
        # #'rsi':rsi.zscore(groupby = sector,mask=rsi.percentile_between(1, 99)),
        #'rsi0': rsi,
        #'rsi1': rsi.zscore(),
        'rsi': rsi.zscore(groupby=sector),
        # 'rsi3': vol20.demean(groupby=sector),
        'market_rank':market.quantiles(100),
        'sector':sector,
        'returns':returns,
        #'ohs':ohs,
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



pd.set_option('display.width', 800)
research = Research()
#print(research.get_engine()._finder)
my_pipe = make_pipeline(research.get_engine()._finder)
result = research.run_pipeline(my_pipe,
                               Date(tz='utc', as_timestamp=True).parser(start),
                               Date(tz='utc', as_timestamp=True).parser(end))
print result
print type(result)
print result.reset_index()
#result.replace([np.inf,-np.inf],np.nan)
result = result.reset_index().drop(['level_0','level_1'],axis = 1).replace([np.inf,-np.inf],np.nan).fillna(0)

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

quadratic_featurizer  = PolynomialFeatures(interaction_only=True)
X_train_quadratic = quadratic_featurizer.fit_transform(result)
print X_train_quadratic
print np.shape(X_train_quadratic),type(X_train_quadratic)

print quadratic_featurizer.get_feature_names(result.columns),len(quadratic_featurizer.get_feature_names(result.columns))

#print type(result.as_matrix()),result.as_matrix