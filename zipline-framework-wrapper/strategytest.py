# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:52:27 2017

@author: kang
"""
import numpy as np
import pandas as pd
import os
from zipline.api import (
    date_rules,
    time_rules,
    pipeline_output,
    record,
    schedule_function,
    symbol,
    get_datetime,
    attach_pipeline,
)
from zipline.pipeline import Pipeline
from me.grocery.strategies.basic_revert_strategy import RevertStrategy
from me.grocery.strategies.basic_factor_strategy import FactorStrategy

from me.grocery.executors.xuqiu_executor import XieqiuExecutor
from me.grocery.riskmanagers.basic_hedge_risk_manager import BasicHedgeRiskManager



from datetime import timedelta, datetime
from me.helper.configure import read_config

def make_pipeline(context):
    columns,universe = context.strategy.pipeline_columns_and_mask()
    return Pipeline(
        columns= columns,
        screen = universe,
    )

def rebalance(context, data):
    print context.pipeline_data
    if (context.sim_params.end_session - get_datetime() > timedelta(days=6)):  # 只在最后一个周末;周5运行
        return
    pipeline_data = context.pipeline_data
    pipeline_data.index = [index.symbol for index in pipeline_data.index]

    shorts, longs = context.strategy.compute_allocation(data,pipeline_data)
    print "to trade:",shorts,longs
    #context.strategy.trade(shorts,longs)
    pass


def __build_strategy(context):
    conf = os.path.dirname('__file__') + './config/global.json'
    config = read_config(conf)
    print "config:",config

    executor = XieqiuExecutor(account=config['account'], password=config['passwd'], portfolio=config['portfolio'])
    executor.login()
    riskmanger = BasicHedgeRiskManager()
    #context.strategy = RevertStrategy(executor, riskmanger)
    context.strategy = FactorStrategy(executor, riskmanger)

def initialize(context):
    __build_strategy(context)
    attach_pipeline(make_pipeline(context), 'my_pipeline')
    schedule_function(rebalance, date_rules.week_end(days_offset=0), half_days=True)  # 周天 ? 周5 ！！！
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
    print "before_trading_start date - %s , price %s" % (get_datetime(), data.current(symbol('000001'), 'price'))
    pass


def recording_statements(context, data):
    # Plot the number of positions over time.
    record(num_positions=len(context.portfolio.positions))