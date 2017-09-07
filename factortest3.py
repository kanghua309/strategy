# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:52:27 2017

@author: kang
"""

from zipline import TradingAlgorithm

from me.pipeline.filters.universe import make_china_equity_universe,default_china_equity_universe_mask



from zipline.api import (
    attach_pipeline,
    date_rules,
    time_rules,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    symbol,
    sid,
    get_datetime,
)

from me.pipeline.factors.boost import Momentum,CrossSectionalReturns

from zipline.pipeline import Pipeline
from zipline.pipeline.factors import RSI
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.data import Column
from zipline.pipeline.data import DataSet
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.pipeline.factors import AverageDollarVolume, CustomFactor, Latest ,RollingLinearRegressionOfReturns
from me.pipeline.classifiers.tushare.sector import get_sector,get_sector_size,get_sector_class
#from me.broker.xuqie.XueqiueLive import login,adjust_weight,get_profolio_position,get_profilio_size,get_profolio_keep_cost_price
from me.broker.xueqiu import XueqiuLive


from me.pipeline.factors.tsfactor import Fundamental


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


MAX_GROSS_LEVERAGE = 1.0
NUM_LONG_POSITIONS = 19 #剔除risk_benchmark
NUM_SHORT_POSITIONS = 0
MAX_BETA_EXPOSURE = 0.20

NUM_ALL_CANDIDATE = NUM_LONG_POSITIONS



MAX_LONG_POSITION_SIZE = 3 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
#MAX_SHORT_POSITION_SIZE = 2*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

MIN_LONG_POSITION_SIZE = 0.5 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

MAX_SECTOR_EXPOSURE = 0.20

risk_benchmark = '000001'
profolio_size = 19
def make_pipeline():

    universe = make_china_equity_universe(
        target_size = 2000,
        mask = default_china_equity_universe_mask([risk_benchmark]),
        max_group_weight= 0.01,
        smoothing_func = lambda f: f.downsample('month_start'),

    )
    last_price = USEquityPricing.close.latest >= 1.0

    universe = universe & last_price

    #universe2 = sector_filter2(100, 0.1)
    #adj = ADV_adj(window_length=252)
    #volume = AverageDollarVolume(window_length=21)
    #cap = MarketCap()
    #liquid = ADV_adj()
    #pred = RNNPredict()
    # Build Filters representing the top and bottom 150 stocks by our combined ranking system.
    # We'll use these as our tradeable universe each day.
    #short = pred.top(1)
    #longs = pred.bottom(1)

    sector = get_sector()
    csreturn = CrossSectionalReturns(window_length=21)
    momentum = Momentum(window_length=21)
    beta = 0.66 * RollingLinearRegressionOfReturns(
        target=symbol(risk_benchmark),  # sid(8554),
        returns_length=6,
        regression_length=21,
        # mask=long_short_screen
        mask=(universe),
    ).beta + 0.33 * 1.0

    combined_rank = (
        momentum.rank(mask=universe).zscore() +
        csreturn.rank(mask=universe).zscore() +
        beta.rank(mask=universe,ascending=False).zscore()
    )
    longs =  combined_rank.top(NUM_ALL_CANDIDATE)
    #shorts = combined_rank.bottom(NUM_SHORT_POSITIONS)
    long_short_screen = (longs)

    beta = 0.66 * RollingLinearRegressionOfReturns(
        target=symbol(risk_benchmark),  # sid(8554),
        returns_length=6,
        regression_length=21,
        # mask=long_short_screen
        mask=(long_short_screen),
    ).beta + 0.33 * 1.0

    return Pipeline(
        columns={
            'longs': longs,
            #'shorts': shorts,
            'combined_rank': combined_rank,
            'momentum': momentum,
            'return' : csreturn,
            'sector': sector.downsample('week_start'),
            'market_beta': beta,
        },
        screen=long_short_screen,
    )


def rebalance(context, data):
    if (context.sim_params.end_session - get_datetime() > timedelta(days=6)):  # 只在最后一个周末;周5运行
        return
    pipeline_data = context.pipeline_data
    pipeline_data.index = [index.symbol for index in pipeline_data.index]
    print "pipeline_data ", len(pipeline_data),pipeline_data
    context.xueqiuLive.login()
    xq_profolio = context.xueqiuLive.get_profolio_keep_cost_price()
    print "Rebalance - Current xq profolio"
    print len(xq_profolio),xq_profolio
    print "Rebalance - To do profolio rebuild   "

    # print "value:", math.log(0.95), math.log(1.05)

    weights = optimalize(context,pipeline_data.index)
    print "Rebalance optimalize weights %s" % weights
    assert not weights.empty


    xq_pos = context.xueqiuLive.get_profolio_position()
    print "xq_pos:",xq_pos
    for stock in xq_pos:
        if stock not in pipeline_data.index:
            print "sell it now ......",stock
            try:
                context.xueqiuLive.adjust_weight(stock, 0)
                pass
            except easytrader.webtrader.TradeError, e:
                print "stock %s trader exception %s" % (stock, e)
                #raise SystemExit(-1)
    for stock,weight in weights.iteritems():
        #weight = df.at[0, c] * 100
        print "stock %s set weight %s" % (stock,weight)
        #if stock == context.xueqiuLive.get_placeholder():
        #    continue #TODO
        try:
            context.xueqiuLive.adjust_weight(stock, weight * 100)
            pass
        except easytrader.webtrader.TradeError as e:
            # except Exception,e:
            print "stock %s trader exception %s" % (stock, e)
            # raise SystemExit(-1)
    pass

def optimalize(context,mask):
    print "mask ;%s" % mask ,type(mask)
    print context.pipeline_data.index
    data = context.pipeline_data.loc[mask]
    print "data \n", data
    todays_universe = data.index
    import cvxpy as cvx
    w = cvx.Variable(len(todays_universe))
    # objective = cvx.Maximize(df.pred.as_matrix() * w)  # mini????
    objective = cvx.Maximize(data.combined_rank.as_matrix() * w)
    constraints = [cvx.sum_entries(w) == 1.0 * MAX_GROSS_LEVERAGE, w >= 0.0]  # dollar-neutral long/short
    # constraints.append(cvx.sum_entries(cvx.abs(w)) <= 1)  # leverage constraint
    constraints.extend([w >= MIN_LONG_POSITION_SIZE, w <= MAX_LONG_POSITION_SIZE])  # long exposure
    riskvec = data.market_beta.fillna(1.0).as_matrix()  # TODO
    constraints.extend([riskvec * w <= MAX_BETA_EXPOSURE])  # risk
    print "MIN_SHORT_POSITION_SIZE %s, MAX_SHORT_POSITION_SIZE %s,MAX_BETA_EXPOSURE %s" % (
    MIN_LONG_POSITION_SIZE, MAX_LONG_POSITION_SIZE, MAX_BETA_EXPOSURE)

    prob = cvx.Problem(objective, constraints)
    prob.solve()
    if prob.status != 'optimal':
        print "Optimal failed %s , do nothing" % prob.status
        return pd.Series()
        # raise SystemExit(-1)
    print np.squeeze(np.asarray(w.value))  # Remove single-dimensional entries from the shape of an array
    return pd.Series(data=np.squeeze(np.asarray(w.value)), index=mask)


def initialize(context):
    attach_pipeline(make_pipeline(), 'my_pipeline')
    schedule_function(rebalance,date_rules.week_end(days_offset=0), half_days=True)  # 周天 ? 周5 ！！！
    # record my portfolio variables at the end of day
    schedule_function(func=recording_statements,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(),
                      half_days=True)

    context.xueqiuLive = XueqiuLive(user = '',account = '18618280998',password = 'Threyear#3',portfolio_code='ZH1140390') #巴颜喀拉山


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