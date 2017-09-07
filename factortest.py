# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:52:27 2017

@author: kang
"""

from zipline import TradingAlgorithm

from me.pipeline.filters.universe import make_china_equity_universe,default_china_equity_universe_mask,private_universe_mask



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



MAX_LONG_POSITION_SIZE = 2 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
#MAX_SHORT_POSITION_SIZE = 2*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

MIN_LONG_POSITION_SIZE = 0.5 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

MAX_SECTOR_EXPOSURE = 0.20

risk_benchmark = '000001'


from me.manager.portfolio import PortfolioManager

def make_pipeline(context):

    universe = make_china_equity_universe(
        target_size = 2000,
        mask = default_china_equity_universe_mask([risk_benchmark]),
        max_group_weight= 0.01,
        smoothing_func = lambda f: f.downsample('month_start'),

    )
    last_price = USEquityPricing.close.latest >= 1.0

    positions = context.xueqiuLive.get_profolio_position() #TODO
    #print positions
    private_universe = private_universe_mask(positions.index)
    universe = universe & last_price | private_universe

    hurst = HurstExp(window_length = int(252*0.25),mask=universe)
    # tradeable_pipe.add(HurstExp().rank(),"hurst_rank")

    '''
    top  = combined_rank.top(50)
    bottom  = combined_rank.bottom(50)
    '''
    top = hurst >= 0.8
    bottom = hurst <= 0.2
    universe = (top | bottom) | private_universe
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
        if keep_price / current_price > 1.10:
            print "%s has down to stop limit, sell it - for %s,%s " % (index,keep_price,current_price)
            stop_dict[index] = 0.0
        if keep_price / current_price < 0.90:
            print "%s has up to expected price , sell it - for %s,%s" % (index,keep_price,current_price)
            stop_dict[index] = 0.0

    return stop_dict

from datetime import timedelta,datetime
def _check_expired_limit(context, data, profolio):
    stop_dict = {}
    for index, value in profolio.iteritems():
        if False == data.can_trade(symbol(index)):
            continue
        lastdt = profolio[index]
        # print "Rebalance - index, keep_price, current_price"
        if datetime.now() - lastdt >  timedelta(weeks=2):
            print "%s has expired , sell it - for %s,%s" % (index,datetime.now(),lastdt)
            stop_dict[index] = 0.0
    return stop_dict


def _adjust_stocks(context,stocks):
    for stock, weight in stocks.iteritems():
        try:
            context.xueqiuLive.adjust_weight(stock,weight * 100)
        except:
            pass

def rebalance(context, data):
    if (context.sim_params.end_session - get_datetime() > timedelta(days=6)):  # 只在最后一个周末;周5运行
        return
    pipeline_data = context.pipeline_data
    pipeline_data.index = [index.symbol for index in pipeline_data.index]
    print "pipeline_data", len(pipeline_data)
    print "---------------------------------"
    #print pipeline_data.loc['000018']
    #context.xueqiuLive.login()
    xq_profolio = context.xueqiuLive.get_profolio_info()
    print "Rebalance - Current xq profolio"
    print len(xq_profolio), xq_profolio

    xq_profolio_real = xq_profolio[xq_profolio['short_time'].isnull()]
    remove_dict = _check_stop_limit(context,data,xq_profolio_real.keep_price)
    print "remove_stock for stop:",remove_dict
    _remove     = _check_expired_limit(context,data,xq_profolio_real.long_time)
    remove_dict.update(_remove) #TODO
    print "remove_stock for expire:",remove_dict

    profolio_hold_index = xq_profolio_real.index.difference(remove_dict)
    print "-----------------------------------sell first------------------------------------------",pipeline_data.ix[profolio_hold_index]
    for index,row in pipeline_data.ix[profolio_hold_index].iterrows():  #应该有很hold里的在data中找不到，没关系，忽略之
        try:
            if False == data.can_trade(symbol(index)):
                continue
        except:  #TODO index 有可能symbol报错
            print "index not exit .........."
            continue
            pass
        hurst = row.hurst
        vbeta = row.volume_pct_beta
        pbeta = row.price_pct_beta
        if  hurst <= 0.2:
            # print("info  sym(%s) is mean revert"%sym)
            if  vbeta > 0 and vbeta < pbeta:
                print("++++++++++++++++++++++++++Info sell sym(%s) for mean revert at all" % index)
                remove_dict[index] = 0.0
        if  hurst >= 0.8:
            # print("info sym(%s) is moment trend"%sym)
            if vbeta < 0 and vbeta > pbeta:
                print("++++++++++++++++++++++++++Info sell sym(%s) for momentum trend at all" % index)
                remove_dict[index] = 0.0

    profolio_hold_index = profolio_hold_index.difference(remove_dict)
    pools = pipeline_data.index.difference(xq_profolio_real.index)
    print "profolio_hold_index before buy:", profolio_hold_index
    print "-----------------------------------buy last------------------------------------------"
    for index,row in pipeline_data.ix[pools].iterrows():
        if False == data.can_trade(symbol(index)):
            continue
        hurst = row.hurst
        vbeta = row.volume_pct_beta
        pbeta = row.price_pct_beta
        if hurst <= 0.2:
            if vbeta < 0 and vbeta < pbeta:  #先买均值回归的！ 安全！！！
                print("==========================Info buy sym(%s) for mean revert" % (index))
                profolio_hold_index = profolio_hold_index.insert(0,index)
        if hurst >= 0.8:
            if vbeta > 0 and vbeta > pbeta:
                print("==========================Info buy sym(%s) for momentum trend" % (index))
                profolio_hold_index = profolio_hold_index.insert(0,index)
        if len(profolio_hold_index) == NUM_LONG_POSITIONS:
            break
        #print "profolio_hold_index:",profolio_hold_index
    print "profolio_hold_index after buy:",profolio_hold_index,len(profolio_hold_index)
    weights = optimalize(context,profolio_hold_index)
    print "profolio weights:", weights


    print "do sell ....."
    #_adjust_stocks(context,remove_dict)
    print "do buy ....."
    #_adjust_stocks(context,weights)
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
    objective = cvx.Maximize(data.market_beta.as_matrix() * w)
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
    context.xueqiuLive = XueqiuLive(user='', account='18618280998', password='Threyear#3',
                                    portfolio_code='ZH1124287')  # 巴颜喀拉山
    context.xueqiuLive.login()
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