#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:52:27 2017

@author: kang
"""

from zipline import TradingAlgorithm
from me.pipeline.classifiers.tushare.sector import getSector
from me.pipeline.filters.universe import universe_filter,sector_filter,sector_filter2


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

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.broadcast size changed")

profolio_size = 10

def make_pipeline():

    universe = sector_filter(100, 0.1)
    universe2 = sector_filter2(100, 0.1)
    print "-----------------------"



    beta = 0.66 * RollingLinearRegressionOfReturns(
        target=symbol('000001'),  # sid(8554),
        returns_length=4,
        regression_length=8,
        #mask=long_short_screen
        mask = (universe|universe2),
    ).beta + 0.33 * 1.0

    #pred = RNNPredict()


    # Build Filters representing the top and bottom 150 stocks by our combined ranking system.
    # We'll use these as our tradeable universe each day.
    #rank  = pred.rank()
    #short = pred.top(1)
    #longs = pred.bottom(1)

    #sector = getSector()


    return Pipeline(
        columns={
            'beta':   beta,
            #'sector': sector,
             #'shorts': test.bottom(2),
        },
    )


def rebalance(context, data):
    print context.pipeline_data.tail(100)
    pass

def initialize(context):
    attach_pipeline(make_pipeline(), 'my_pipeline')
    schedule_function(rebalance,date_rules.week_start(days_offset=0),half_days = True) #每周一
    pass

def handle_data(context, data):
    pass

def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')
    #print "date - %s , price %s" % (get_datetime(),data.current(symbol('000001'), 'price'))
    pass
