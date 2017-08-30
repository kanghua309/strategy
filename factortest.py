#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:52:27 2017

@author: kang
"""

from zipline import TradingAlgorithm

from me.pipeline.filters.universe import make_china_equity_universe
from me.pipeline.factors.tsfactor import default_china_equity_universe_mask



from zipline.api import (
    attach_pipeline,
    date_rules,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    symbol,
    sid,
    get_datetime,

)

from zipline.pipeline import Pipeline
from zipline.pipeline.factors import RSI
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.data import Column  
from zipline.pipeline.data import DataSet
from zipline.pipeline.engine import SimplePipelineEngine  
from zipline.pipeline.loaders.frame import DataFrameLoader 
from zipline.pipeline.factors import AverageDollarVolume, CustomFactor, Latest ,RollingLinearRegressionOfReturns

from me.pipeline.factors.prediction import RNNPredict

from   itertools import chain
import numpy as np
#import pandas.io.data as web
import pandas_datareader.data as web
#from   pandas.stats.api import ols
import pandas as pd
#import math
#import pytz
from datetime import timedelta, date, datetime
import easytrader


profolio_size = 10

def make_pipeline():

    #universe = sector_filter(100, 0.1).downsample('month_start')
    universe = make_china_equity_universe(
        target_size = 1000,
        mask = default_china_equity_universe_mask(),
        max_group_weight= 0.3,
        smoothing_func = lambda f: f.downsample('month_start')
    )

    #universe2 = sector_filter2(100, 0.1)
    print "-----------------------"

    beta = 0.66 * RollingLinearRegressionOfReturns(
        target=symbol('000001'),  # sid(8554),
        returns_length=4,
        regression_length=8,
        #mask=long_short_screen
        mask = (universe),
    ).beta + 0.33 * 1.0


    #liquid = ADV_adj()
    #pred = RNNPredict()


    # Build Filters representing the top and bottom 150 stocks by our combined ranking system.
    # We'll use these as our tradeable universe each day.
    #rank  = pred.rank()
    #short = pred.top(1)
    #longs = pred.bottom(1)

    #sector = getSector()


    return Pipeline(
        columns={
            'beta': beta.downsample('month_start'),
            #'adv' : liquid.downsample('month_start'),
            #'sector': sector,
             #'shorts': test.bottom(2),
        },
        screen=universe,
    )


def rebalance(context, data):

    print "rebalance ----",context.pipeline_data.tail(5)
    pass

def initialize(context):
    attach_pipeline(make_pipeline(), 'my_pipeline')
    schedule_function(rebalance,date_rules.week_start(days_offset=0),half_days = True) #每周一
    pass

def handle_data(context, data):
    print "handle_data %s" % (get_datetime())
    pass

def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')
    print "before_trading_start date - %s , price %s" % (get_datetime(),data.current(symbol('000001'), 'price'))
    pass
