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

import talib

class ILLIQ(CustomFactor):
    inputs = [USEquityPricing.close,USEquityPricing.volume]
    window_length = int(252)
    def compute(self, today, assets, out, close,volume):
        window_length = len(close)
        #print "window length",window_length
        _rets = np.abs(pd.DataFrame(close, columns=assets).pct_change()[1:])
        #print(_rets.head(10))
        _vols  = pd.DataFrame(volume, columns= assets)[1:]
        #print(_vols.head(10))
        #print "--------------------------1"
        #print(_rets/_vols)
        #print "--------------------------2"
        #print pd.rolling_mean(_rets/_vols, window=window_length-1)
        #print (_rets/_vols).mean(),type((_rets/_vols).mean())
        out[:] =(_rets/_vols).mean().values


# class MOM(CustomFactor):
#     # this class generates the MACD as a Percentage
#     inputs = [USEquityPricing.close]
#     window_length = int(252)
#
#     def columnwise_anynan(self,array2d):
#         return np.isnan(array2d).any(axis=0)
#     def compute(self, today, assets, out, close):
#             window_length= len(close)
#             print window_length,np.shape(close)
#             anynan = self.columnwise_anynan(close)
#             for col_ix, have_nans in enumerate(anynan):
#                 if have_nans:
#                     out[col_ix] = np.nan
#                     continue
#                 print(window_length,close[:, col_ix])
#                 mom = talib.MOM(close[:, col_ix],timeperiod=window_length-1)
#
#                 out[col_ix] = mom[-1]



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
end   = '2015-9-8'  # 必须在国内交易日

def make_pipeline(asset_finder):
    # h2o = USEquityPricing.high.latest / USEquityPricing.open.latest
    # l2o = USEquityPricing.low.latest / USEquityPricing.open.latest
    # c2o = USEquityPricing.close.latest / USEquityPricing.open.latest
    # h2c = USEquityPricing.high.latest / USEquityPricing.close.latest
    # l2c = USEquityPricing.low.latest / USEquityPricing.close.latest
    # h2l = USEquityPricing.high.latest / USEquityPricing.low.latest
    #
    # vol = USEquityPricing.volume.latest
    # outstanding = Fundamental(asset_finder).outstanding
    # outstanding.window_safe = True
    # turnover_rate = vol / Latest([outstanding])
    # returns = Returns(inputs=[USEquityPricing.close], window_length=5)  # 预测一周数据
    private_universe = private_universe_mask(['000001','000002'],asset_finder=asset_finder)

    illiq = ILLIQ(window_length=20,mask = private_universe)
    ep = 1/Fundamental(asset_finder).pe
    bp = 1/Fundamental(asset_finder).pb
    bvps = Fundamental(asset_finder).bvps
    rev20 = Returns(inputs=[USEquityPricing.close], window_length=20)
    vol20 = AverageDollarVolume(window_length=20)
    rsi = RSI(window_length=20)

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
        'vol20':vol20,
        'rsi':rsi,
    }
    # pipe_screen = (low_returns | high_returns)
    pipe = Pipeline(columns=pipe_columns,screen=private_universe)
    return pipe



pd.set_option('display.width', 800)
research = Research()
#print(research.get_engine()._finder)
my_pipe = make_pipeline(research.get_engine()._finder)
result = research.run_pipeline(my_pipe,
                               Date(tz='utc', as_timestamp=True).parser(start),
                               Date(tz='utc', as_timestamp=True).parser(end))
print result