# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:42:21 2017

@author: kanghua
"""
import numpy as np
import pandas as pd

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor, Returns, Latest
from me.pipeline.factors.tsfactor import Fundamental


class ILLIQ(CustomFactor):
    inputs = [USEquityPricing.close,USEquityPricing.volume]
    window_length = int(252)
    def compute(self, today, assets, out, close,volume):
        _rets = pd.DataFrame(close, columns=assets).pct_change()[1:]
        print(_rets.head(10))
        _vols  = pd.DataFrame(volume, columns= assets)[1:]
        print(_vols.head(10))
        print "--------------------------1"
        print(_rets/_vols)
        print "--------------------------2"

        print pd.rolling_mean(_rets/_vols, window=2)
        #out[:] =


import click
import numpy as np
import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Latest
from zipline.pipeline.factors import Returns
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

    illiq = ILLIQ(window_length=10,mask = private_universe)
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
    }
    # pipe_screen = (low_returns | high_returns)
    pipe = Pipeline(columns=pipe_columns,screen=private_universe)
    return pipe



pd.set_option('display.width', 800)
research = Research()
print(research.get_engine()._finder)
my_pipe = make_pipeline(research.get_engine()._finder)
result = research.run_pipeline(my_pipe,
                               Date(tz='utc', as_timestamp=True).parser(start),
                               Date(tz='utc', as_timestamp=True).parser(end))