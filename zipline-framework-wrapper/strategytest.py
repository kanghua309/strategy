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
from me.grocery.strategies.basic_dl_strategy import DLStrategy

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
    #print context.pipeline_data
    if (context.sim_params.end_session - get_datetime() > timedelta(days=6)):  # 只在最后一个周末;周5运行
        return
    pipeline_data = context.pipeline_data
    pipeline_data.index = [index.symbol for index in pipeline_data.index]

    shorts, longs = context.strategy.compute_allocation(data,pipeline_data)
    print "to trade:",shorts,longs
    #context.strategy.trade(shorts,longs)
    pass


def __build_revert_basic_strategy(context):
    conf = os.path.dirname('__file__') + './config/global.json'
    config = read_config(conf)
    print "config:",config
    executor = XieqiuExecutor(account=config['account'], password=config['passwd'], portfolio=config['portfolio'])
    executor.login()
    #riskmanger = BasicHedgeRiskManager()
    riskmanger = BasicHedgeRiskManager()
    context.strategy = RevertStrategy(executor, riskmanger)

def __build_factor_basic_strategy(context):
    conf = os.path.dirname('__file__') + './config/global.json'
    config = read_config(conf)
    print "config:",config
    executor = XieqiuExecutor(account=config['account'], password=config['passwd'], portfolio=config['portfolio'])
    executor.login()
    #riskmanger = BasicHedgeRiskManager()
    riskmanger = BasicHedgeRiskManager()
    context.strategy = FactorStrategy(executor, riskmanger,str(context.sim_params.end_session)[:10]) #最后一天触发预测

def __build_deeplearn_strategy(context):
    conf = os.path.dirname('__file__') + './config/global.json'
    config = read_config(conf)
    print "config:",config
    executor = XieqiuExecutor(account=config['account'], password=config['passwd'], portfolio=config['portfolio'])
    executor.login()
    #riskmanger = BasicHedgeRiskManager()
    riskmanger = BasicHedgeRiskManager()
    context.strategy = DLStrategy(executor, riskmanger,str(context.sim_params.end_session)[:10])


def initialize(context):
    #__build_revert_basic_strategy(context)
    #__build_factor_basic_strategy(context)
    __build_deeplearn_strategy(context)
    attach_pipeline(make_pipeline(context), 'my_pipeline')
    schedule_function(rebalance, date_rules.week_end(days_offset=0), half_days=True)  # 周天 ? 周5 ！！！
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)
    print "initialize over"
    pass


def handle_data(context, data):
    print "handle day:%s" % (get_datetime())
    pass


def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')
    pass


def recording_statements(context, data):
    # Plot the number of positions over time.
    record(num_positions=len(context.portfolio.positions))