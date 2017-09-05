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
#from me.broker.xuqie.XueqiueLive import login,adjust_weight,get_profolio_position,get_profilio_size,get_profolio_keep_cost_price
from me.pipeline.factors.boost import HurstExp,Beta

from me.broker.xueqiu import XueqiuLive


from me.pipeline.factors.tsfactor import Fundamental
from   itertools import chain
import numpy as np
import pandas as pd
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


from me.manager.portfolio import PortfolioManager

def make_pipeline():

    universe = make_china_equity_universe(
        target_size = 2000,
        mask = default_china_equity_universe_mask([risk_benchmark]),
        max_group_weight= 0.01,
        smoothing_func = lambda f: f.downsample('month_start'),

    )
    last_price = USEquityPricing.close.latest >= 1.0
    universe = universe & last_price

    hurst = HurstExp(window_length = int(252*0.25),mask=universe)
    # tradeable_pipe.add(HurstExp().rank(),"hurst_rank")

    '''
    top  = combined_rank.top(10)
    bottom  = combined_rank.bottom(10)
    '''
    top = hurst >= 0.8
    bottom = hurst <= 0.2
    universe = (top | bottom)
    combined_rank = (
        hurst.rank(mask=universe)
    )

    #sector = get_sector
    pct_beta = Beta(window_length=21,mask=(universe))
    risk_beta = 0.66 * RollingLinearRegressionOfReturns(
        target=symbol(risk_benchmark),  # sid(8554),
        returns_length=6,
        regression_length=21,
        # mask=long_short_screen
        mask=(universe),
    ).beta + 0.33 * 1.0

    return Pipeline(
        columns={
            'hurst': hurst.downsample('week_start'),
            'price_pct_beta' : pct_beta.pbeta,
            'volume_pct_beta': pct_beta.vbeta,
            #'sector': sector.downsample('week_start'),
            'market_beta': risk_beta,
            'rank':combined_rank,
            #'testrank':hurst.rank(mask=universe)
        },
        screen=universe,
    )


def _check_stop_limit(context,data,profolio):
    stop_dict = {}
    for index,value in profolio.iteritems():
        if False == data.can_trade(symbol(index)):
            continue
        keep_price = profolio[index]
        current_price = data.current(symbol(index), 'price')
        #print "Rebalance - index, keep_price, current_price"
        if keep_price / current_price > 1.15:
            print "%s has down to stop limit, sell it - for %s,%s " % (index,keep_price,current_price)
            stop_dict[index] = True
        if keep_price / current_price < 0.90:
            print "%s has up to expected price , sell it - for %s,%s" % (index,keep_price,current_price)
            stop_dict[index] = True
    return stop_dict

def rebalance(context, data):
    if (context.sim_params.end_session - get_datetime() > timedelta(days=6)):  # 只在最后一个周末;周5运行
        return
    pipeline_data = context.pipeline_data
    pipeline_data.index = [index.symbol for index in pipeline_data.index]
    print "pipeline_data", len(pipeline_data)
    print pipeline_data.head(70)
    print "---------------------------------"
    #print pipeline_data.loc['000018']
    context.xueqiuLive.login()
    xq_profolio = context.xueqiuLive.get_profolio_keep_cost_price()
    print "Rebalance - Current xq profolio"
    print len(xq_profolio), xq_profolio
    remove_dict = _check_stop_limit(context,data,xq_profolio)
    print "remove_stock:",remove_dict
    profolio_hold_index = xq_profolio.index.difference(remove_dict)
    print  profolio_hold_index
    print "-----------------------------------sell first------------------------------------------",pipeline_data.ix[profolio_hold_index]
    for index,row in pipeline_data.ix[profolio_hold_index].iterrows():  #应该有很hold里的在data中找不到，没关系，忽略之
        hurst = row.hurst
        vbeta = row.volume_pct_beta
        pbeta = row.price_pct_beta
        if  hurst <= 0.2:
            # print("info  sym(%s) is mean revert"%sym)
            if  vbeta > 0 and vbeta < pbeta:
                print("++++++++++++++++++++++++++Info sell sym(%s) for mean revert at all" % index)
                remove_dict[index] = True
        if hurst >= 0.8:
            # print("info sym(%s) is moment trend"%sym)
            if vbeta < 0 and vbeta > pbeta:
                print("++++++++++++++++++++++++++Info sell sym(%s) for momentum trend at all" % index)
                remove_dict[index] = True

    profolio_hold_index = profolio_hold_index.difference(remove_dict)
    pools = pipeline_data.index.difference(xq_profolio.index)
    print "profolio_hold_index:", profolio_hold_index
    print "-----------------------------------buy last------------------------------------------",pools
    for index,row in pipeline_data.ix[pools].iterrows():
        hurst = row.hurst
        vbeta = row.volume_pct_beta
        pbeta = row.price_pct_beta
        if hurst <= 0.2:
            if vbeta < 0 and vbeta < pbeta:
                print("==========================Info buy sym(%s) for mean revert" % (index))
                profolio_hold_index = profolio_hold_index.insert(-1,index)
        if hurst >= 0.8:
            if vbeta > 0 and vbeta > pbeta:
                print("==========================Info buy sym(%s) for momentum trend" % (index))
                profolio_hold_index = profolio_hold_index.insert(-1,index)
        print "profolio_hold_index:",profolio_hold_index

    weights = optimalize(context,profolio_hold_index)

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
    objective = cvx.Maximize(data.rank.as_matrix() * w)
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

    context.xueqiuLive = XueqiuLive(user = '',account = '18618280998',password = 'Threeeyear3#',portfolio_code='ZH1140390') #巴颜喀拉山
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