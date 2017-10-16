# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:52:27 2017

@author: kang
"""
import os
from zipline.api import (
    date_rules,
    time_rules,
    pipeline_output,
    schedule_function,
    get_datetime,
    attach_pipeline,
)
from zipline.pipeline import Pipeline

from campaign.deeplearn.xuqiu_dl_execute_strategy import DLExampleStrategy
from me.grocery.executors.xuqiu_executor import XieqiuExecutor
from me.grocery.riskmanagers.basic_hedge_risk_manager import BasicHedgeRiskManager
from me.helper.configure import read_config


def make_pipeline(context):
    columns,universe = context.strategy.pipeline_columns_and_mask()
    return Pipeline(
        columns= columns,
        screen = universe,
    )

def rebalance(context, data):
    #print context.pipeline_data
    print ("today 0 :", type(get_datetime()), get_datetime(), type(context.sim_params.end_session),
           context.sim_params.end_session.day)
    if (context.sim_params.end_session.day != get_datetime().day):  # 只在最后一个周一开盘前运行
        return
    print ("today 1 :", get_datetime())

    pipeline_data = context.pipeline_data
    pipeline_data.index = [index.symbol for index in pipeline_data.index]

    shorts, longs = context.strategy.compute_allocation(data,pipeline_data)
    print ("to trade:",shorts,longs)
    context.strategy.trade(shorts,longs)
    pass


def __build_deeplearn_strategy(context):
    conf = os.path.dirname('__file__') + './config/global.json'
    config = read_config(conf)
    print ("config:",config)
    executor = XieqiuExecutor(account=config['account'], password=config['passwd'], portfolio=config['portfolio'])
    executor.login()
    #riskmanger = BasicHedgeRiskManager()
    riskmanger = BasicHedgeRiskManager()
    context.strategy = DLExampleStrategy(executor, riskmanger,str(context.sim_params.end_session)[:10])

def initialize(context):

    __build_deeplearn_strategy(context)
    attach_pipeline(make_pipeline(context), 'my_pipeline')
    schedule_function(rebalance, date_rules.every_day(), time_rules.every_minute())  # 每天调度，但只有最后一天运行
    # record my portfolio variables at the end of day
    print ("initialize over")
    pass


def handle_data(context, data):
    print ("handle day:%s" % (get_datetime()))
    pass


def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')
    pass
