# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:52:27 2017

@author: kang
"""

import numpy as np
import pandas as pd
from zipline.api import (
    attach_pipeline,
    date_rules,
    time_rules,
    pipeline_output,
    record,
    schedule_function,
    symbol,
    get_datetime,
)
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import RollingLinearRegressionOfReturns

from me.pipeline.factors.alpha101 import Alpha48
from me.pipeline.factors.pattern import PatternFactor
from me.grocery.broker.xueqiu import XueqiuLive
from me.pipeline.classifiers.tushare.sector import get_sector
# from me.broker.xuqie.XueqiueLive import login,adjust_weight,get_profolio_position,get_profilio_size,get_profolio_keep_cost_price
from me.pipeline.factors.boost import HurstExp, Beta
from me.pipeline.filters.universe import make_china_equity_universe, default_china_equity_universe_mask, \
    private_universe_mask
from datetime import timedelta, datetime

MAX_GROSS_LEVERAGE = 1.0
NUM_LONG_POSITIONS = 19 #剔除risk_benchmark
NUM_SHORT_POSITIONS = 0
MAX_BETA_EXPOSURE = 0.20

NUM_ALL_CANDIDATE = NUM_LONG_POSITIONS



MAX_LONG_POSITION_SIZE = 5 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
#MAX_SHORT_POSITION_SIZE = 2*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

MIN_LONG_POSITION_SIZE = 0.1 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

MAX_SECTOR_EXPOSURE = 0.40

risk_benchmark = '000001'


def make_pipeline(context):

    universe = make_china_equity_universe(
        target_size = 2000,
        mask = default_china_equity_universe_mask([risk_benchmark]),
        max_group_weight= 0.01,
        smoothing_func = lambda f: f.downsample('month_start'),

    )

    sector = get_sector()


    alpha48 = Alpha48()

    pattern = PatternFactor(window_length = 42, indentification_lag=1)
    return Pipeline(
        columns={
            #'sector': sector.downsample('week_start'),
            #'alpha48':alpha48,
            'pattern':pattern,
            #'testrank':hurst.rank(mask=universe)
        },
        screen=universe,
    )



def rebalance(context, data):
    print "pipeline_data",context.pipeline_data
    if (context.sim_params.end_session - get_datetime() > timedelta(days=6)):  # 只在最后一个周末;周5运行
        return
    pipeline_data = context.pipeline_data
    pipeline_data.index = [index.symbol for index in pipeline_data.index]
    print "pipeline_data", len(pipeline_data)

    pass

def optimalize(context,mask):
    pass


def initialize(context):
    #context.xueqiuLive = XueqiuLive(user='', account='18618280998', password='Threyear#3',
    #                                portfolio_code='ZH1124287')  # 巴颜喀拉山
    #context.xueqiuLive.login()
    attach_pipeline(make_pipeline(context), 'my_pipeline')
    schedule_function(rebalance,date_rules.week_end(days_offset=0), half_days=True)  # 周天 ? 周5 ！！！
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)
    pass

def handle_data(context, data):
    print "handle_data %s" % (get_datetime())
    pass

def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')
    print "before_trading_start date - %s , price %s" % (get_datetime(),data.current(symbol('000001'), 'price'))
    pass

def recording_statements(context, data):
    # Plot the number of positions over time.
    record(num_positions=len(context.portfolio.positions))