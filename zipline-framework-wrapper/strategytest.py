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
from me.grocery.strategies.xuqiu_basic_revert_strategy import RevertStrategy
from me.grocery.strategies.basic_factor_strategy_example import BasicFactorStrategy
from me.grocery.strategies.xuqiu_basic_dl_strategy import DLStrategy

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

    pipeline_data = context.pipeline_data
    pipeline_data.index = [index.symbol for index in pipeline_data.index]

    shorts, longs = context.strategy.compute_allocation(data,pipeline_data)
    print ("to trade:",shorts,longs)
    context.strategy.trade(shorts,longs)
    pass




def __build_factor_basic_strategy(context):
    riskmanger = BasicHedgeRiskManager()
    context.strategy = BasicFactorStrategy(riskmanger)

def initialize(context):
    __build_factor_basic_strategy(context)
    attach_pipeline(make_pipeline(context), 'my_pipeline')
    schedule_function(rebalance, date_rules.week_end(days_offset=0), half_days=True)
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)
    print ("initialize over")
    pass


def handle_data(context, data):
    print ("handle day:%s" % (get_datetime()))
    pass


def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')
    pass


def recording_statements(context, data):
    # Plot the number of positions over time.
    record(num_positions=len(context.portfolio.positions))