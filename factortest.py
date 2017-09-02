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
    time_rules,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    symbol,
    sid,
    get_datetime,
)

from me.pipeline.factors.tsfactor import default_china_equity_universe_mask
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
NUM_LONG_POSITIONS  = 20
NUM_SHORT_POSITIONS = 0
MAX_BETA_EXPOSURE = 0.30

MAX_LONG_POSITION_SIZE = 4 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
#MAX_SHORT_POSITION_SIZE = 2*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

MIN_LONG_POSITION_SIZE = 0.5 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

MAX_SECTOR_EXPOSURE = 0.30


profolio_size = 19
def make_pipeline():

    universe = make_china_equity_universe(
        target_size = 2000,
        mask = default_china_equity_universe_mask(),
        max_group_weight= 0.01,
        smoothing_func = lambda f: f.downsample('month_start'),

    )

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
    combined_rank = (
        momentum.rank(mask=universe).zscore() +
        csreturn.rank(mask=universe).zscore()
    )
    longs =  combined_rank.top(NUM_LONG_POSITIONS)
    #shorts = combined_rank.bottom(NUM_SHORT_POSITIONS)
    long_short_screen = (longs)

    beta = 0.66 * RollingLinearRegressionOfReturns(
        target=symbol('000001'),  # sid(8554),
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
            'sector': sector,
            'market_beta': beta
        },
        screen=long_short_screen,
    )


def rebalance(context, data):
    pipeline_data = context.pipeline_data
    print "rebalance ----",len(context.pipeline_data),get_datetime()
    #print "describe adj :\n", context.pipeline_data.adj.describe()
    #print "describe volume:\n", context.pipeline_data.volume.describe()
    #print "describe cap:\n", context.pipeline_data.cap.describe()
    print "data \n", pipeline_data
    todays_universe = pipeline_data.index
    ### Extract from pipeline any specific risk factors you want
    # to neutralize that you have already calculated
    #risk_factor_exposures = pd.DataFrame({
    #    'market_beta':pipeline_data.market_beta.fillna(1.0)
    #})


    print "Rebalance - now new profolio"
    import cvxpy as cvx

    w = cvx.Variable(len(todays_universe))
    # objective = cvx.Maximize(df.pred.as_matrix() * w)  # mini????
    objective = cvx.Maximize(pipeline_data.combined_rank.as_matrix() * w)

    constraints = [cvx.sum_entries(w) == 1.0*MAX_GROSS_LEVERAGE, w >= 0.0]  # dollar-neutral long/short
    # constraints.append(cvx.sum_entries(cvx.abs(w)) <= 1)  # leverage constraint
    constraints.extend([w >= MIN_LONG_POSITION_SIZE, w <= MAX_LONG_POSITION_SIZE])  # long exposure
    riskvec = pipeline_data.market_beta.fillna(1.0).as_matrix() #TODO

    constraints.extend([riskvec * w <= MAX_BETA_EXPOSURE])  # risk

    print "MIN_SHORT_POSITION_SIZE %s, MAX_SHORT_POSITION_SIZE %s,MAX_BETA_EXPOSURE %s" %(MIN_LONG_POSITION_SIZE,MAX_LONG_POSITION_SIZE,MAX_BETA_EXPOSURE)

    # filters = [i for i in range(len(africa)) if africa[i] == 1]
    #版块对冲当前，因为股票组合小，不合适
    '''
    sector_dict = {}
    idx = 0
    #print pipeline_data.sector
    for equite,classid in pipeline_data.sector.iteritems():
        #print("--------###", equite.symbol, classid)
        if classid not in sector_dict:
            _ = []
            sector_dict[classid] = _
        sector_dict[classid].append(idx)
        idx += 1
    sector_size = len(sector_dict)
    for k, v in sector_dict.iteritems():
        print sector_size,v ,1.0/sector_size * ( 1.0 + MAX_SECTOR_EXPOSURE) ,1.0/sector_size *( 1.0 - MAX_SECTOR_EXPOSURE)
        constraints.append(cvx.sum_entries(w[v]) <= 1.0/sector_size * ( 1.0+ MAX_SECTOR_EXPOSURE)),
        constraints.append(cvx.sum_entries(w[v]) >= 1.0/sector_size * ( 1.0 - MAX_SECTOR_EXPOSURE))
    '''
    # print("risk_factor_exposures.as_matrix().T",pipeline_data.market_beta.fillna(1.0),pipeline_data.market_beta.fillna(1.0).values)
    # constraints.append(pipeline_data.market_beta.fillna(1.0)*w<= MAX_BETA_EXPOSURE)

    prob = cvx.Problem(objective, constraints)
    prob.solve()
    if prob.status != 'optimal':
        print "Optimal failed %s , do nothing" % prob.status
        return
        #raise SystemExit(-1)

    print np.squeeze(np.asarray(w.value))  # Remove single-dimensional entries from the shape of an array
    pass

def initialize(context):
    attach_pipeline(make_pipeline(), 'my_pipeline')
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