#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:52:27 2017

@author: kang
"""

from zipline import TradingAlgorithm

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
from zipline.pipeline.factors import AverageDollarVolume, CustomFactor, Latest, RollingLinearRegressionOfReturns

from me.pipeline.factors.prediction import RNNPredict
from me.pipeline.classifiers.tushare.sector import getSector
from me.pipeline.filters.universe import universe_filter,sector_filter

import math

from itertools import chain
import numpy as np
# import pandas.io.data as web
import pandas_datareader.data as web
# from  pandas.stats.api import ols
import pandas as pd
# import math
# import pytz
from datetime import timedelta, date, datetime
import easytrader

profolio_size = 19  ##FIX  IT
exchange = 'xq'
user = ''
account = '18618280998'
password = 'Threeeyear3#'
portfolio_code='ZH1135253'


def make_pipeline():
    beta = 0.66 * RollingLinearRegressionOfReturns(
        target=symbol('000001'),  # sid(8554),
        returns_length=5,
        regression_length=25,
        # mask=long_short_screen
    ).beta + 0.33 * 1.0

    pred = RNNPredict()
    # Build Filters representing the top and bottom 150 stocks by our combined ranking system.
    # We'll use these as our tradeable universe each day.
    rank = pred.rank()
    #short = pred.top(1)
    #longs = pred.bottom(1)

    #sector = getSector()


    return Pipeline(
        columns={
            'pred': pred,
            'rank': rank,
            'beta': beta,
            #'sector': sector,
            # 'longs': longs,
           # 'short': short
            # 'shorts': test.bottom(2),
        },
    )


#怎么避开没开盘的股票呢？
def rebalance(context, data):

    #print data.can_trade(symbol('600701')),data.can_trade(symbol('000806'))
    # Pipeline data will be a dataframe with boolean columns named 'longs' and
    # 'shorts'.
    #print context.sim_params.end_session, type(context.sim_params.end_session), type(get_datetime())

    #print("===============================================debug print rebalance TODAY:", get_datetime(), context.sim_params.end_session - get_datetime())
    if (context.sim_params.end_session - get_datetime() > timedelta(days=6)): #只在最后一个周末运行
        return
    print ""
    print "Rebalance - ToDay(%s),AssetSize（%d)" % (get_datetime(),len(context.pipeline_data.index))
    pipeline_data = context.pipeline_data
    all_assets = pipeline_data.index
    print "Rebalance - To do profolio rebuild   "
    df = pipeline_data.sort_values(axis=0, by='pred', ascending=False)
    df.index = [index.symbol for index in df.index]
    print "Rebalance - Pipeline factor"
    print (df.head(10))
    xq_profolio = get_xq_profolio_keep_cost_price(context)
    print "Rebalance - Current xq profolio"
    print xq_profolio
    print "Rebalance - To do profolio rebuild   "

    #print "value:", math.log(0.95), math.log(1.05)
    xq_new_profolio_dict = xq_profolio.to_dict()
    print "Rebalance - Step1 - check which stock should sell"
    for index, value in xq_profolio.iteritems():
        if False == data.can_trade(symbol(index)):
            continue
        #print index, value
        keep_price = xq_profolio[index]
        current_price = data.current(symbol(index), 'price')
        print "Rebalance - index, keep_price, current_price"
        if keep_price / current_price > 1.05:
            print "%s has down to stop limit, sell it" % index
            del xq_new_profolio_dict[index]
        if keep_price / current_price < 0.95:
            print "%s has up to expected price , sell it" % index
            del xq_new_profolio_dict[index]
            # if df.at[index,"pred"] > math.log(1.05):
            #    print "it predict will down , sell it", index,df.at[index,"pred"]
            #    del xq_new_profolio_dict[index]

    df = df.sort_values(axis=0, by='pred', ascending=True)
    xqdf = df.ix[xq_new_profolio_dict].sort_values(axis=0, by='pred', ascending=False)  # 给定集合的行
    print "Rebalance - now new profolio"
    print xq_new_profolio_dict

    # xq_profolio = get_xq_profolio()
    print "Rebalance - Step2 - check which stock should buy"
    slotlen = len(xq_new_profolio_dict)
    freeslotlen = profolio_size - slotlen
    print "Rebalance - free slot count %s" % (profolio_size - slotlen)

    for index1, row1 in df.iterrows():  # 获取每行的index、row
        if False == data.can_trade(symbol(index1)):
            continue
        if freeslotlen > 0 and not xq_new_profolio_dict.has_key(index1):
            xq_new_profolio_dict[index1] = 0
            freeslotlen -= 1
            continue  # 先用最高预测吧slot空填了
        if index1 == '000001':
            continue
        # 在检查还有没有可替换的东东
        for index2, row2 in xqdf.iterrows():
            if False == data.can_trade(symbol(index2)):
                continue
            if row1["pred"] < math.log(0.95) and not xq_new_profolio_dict.has_key(
                    index1) and xq_new_profolio_dict.has_key(index2):
                print "Rebalance - it predict will up", index1, row1["pred"]
                if abs(row1["pred"] - row2["pred"]) > math.log(0.05):
                    print "Rebalance - instead my stock %s in profolio by %s" % (index2,index1)
                    print xq_new_profolio_dict[index2]
                    del xq_new_profolio_dict[index2]
                    xq_new_profolio_dict[index1] = 0
                break

    print "Rebalance - now new profolio"
    print xq_new_profolio_dict




    print "Rebalance - Step3 - To do cvx optimse new  profolio weight"
    import cvxpy as cvx
    print "Rebalance - To do cvx optimse in our new profolio(%s) set" % len(xq_new_profolio_dict)
    #print xq_new_profolio_dict
    df = df.ix[xq_new_profolio_dict]
    print df.head(10)
    w = cvx.Variable(len(df.index))
    objective = cvx.Maximize(df.pred.as_matrix() * w)  # mini????

    constraints = [cvx.sum_entries(w) == 1, w > 0]  # dollar-neutral long/short
    # constraints.append(cvx.sum_entries(cvx.abs(w)) <= 1)  # leverage constraint
    constraints.extend([w > 0.05, w <= 0.2])  # long exposure
    riskvec = df.beta.fillna(1.0).as_matrix()
    MAX_BETA_EXPOSURE = 0.20
    constraints.extend([riskvec * w <= MAX_BETA_EXPOSURE])  # risk

    # filters = [i for i in range(len(africa)) if africa[i] == 1]
    '''  #版块对冲当前，因为股票组合小，不合适
    secMap = {}
    idx = 0
    #print pipeline_data.sector
    for equite,classid in df.sector.iteritems():
        #print("--------###", equite.symbol, classid)
        if classid not in secMap:
            _ = []
            secMap[classid] = _
        secMap[classid].append(idx)
        idx += 1
    for k, v in secMap.iteritems():
        print(len(secMap))
        print(w[v])
        constraints.append(cvx.sum_entries(w[v]) == 1/len(secMap))
    '''
    # print("risk_factor_exposures.as_matrix().T",pipeline_data.market_beta.fillna(1.0),pipeline_data.market_beta.fillna(1.0).values)
    # constraints.append(pipeline_data.market_beta.fillna(1.0)*w<= MAX_BETA_EXPOSURE)

    prob = cvx.Problem(objective,constraints)
    prob.solve()
    if prob.status != 'optimal':
       print "optimal failed ", prob.status
       raise SystemExit(-1)

    print np.squeeze(np.asarray(w.value))  # Remove single-dimensional entries from the shape of an array

    print "Rebalance - Step4 - do real trading "
    print "Rebalance - sell some first"
    xq_pos = get_xq_pos(context)
    for stock in xq_pos:
        if stock not in xq_new_profolio_dict:
            print "sell it now ......"
            try:
                 context.user.adjust_weight(stock,0)
            except easytrader.webtrader.TradeError, e:
                print "Trader exception %s", e
                raise SystemExit(-1)

    print "Rebalance - adjust some then"
    df = pd.DataFrame(data=np.transpose((w.value)), columns=df.index)  # 翻转
    for c in df.columns:
        weight = df.at[0,c] * 100
        print "stock %s set weight %s" %(c,weight)
        try:
            context.user.adjust_weight(c,weight)
            pass
        except easytrader.webtrader.TradeError,e:
            print "Trader exception %s" % e
            raise SystemExit(-1)


def initialize(context):
    context.user = easytrader.use(exchange)
    context.user.prepare(user=user, account=account, password=password, portfolio_code=portfolio_code)
    attach_pipeline(make_pipeline(), 'my_pipeline')
    schedule_function(rebalance, date_rules.week_end(days_offset=0), half_days=True)  # 周天


def handle_data(context, data):
    print ".",
    #print "date - %s , price %s" % (get_datetime(),data.current(symbol('600701'), 'price'))
    #print "date - %s , price %s" % (get_datetime(),data.current(symbol('603030'), 'price'))
    pass


def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')
    pass

'''
def sell_xq_all(context):
    #xq_user = easytrader.use('xq')
    #xq_user.prepare(user='', account='18618280998', password='Threeeyear3#', portfolio_code='ZH1135253')
    for pos in context.xq_user.position:
        print pos
        print pos['stock_code'][2:]
        if (pos['stock_code'][2:] != '000001'):
            context.user.adjust_weight(pos['stock_code'][2:], 0.0)


def sell_xq(context,stock):
    context.user.adjust_weight(stock, 0.0)


def sell_batch_xq(context,stocks):
    for stock in stocks:
        context.user.adjust_weight(stock, 0.0)
'''
def get_xq_pos(context):
    df = pd.DataFrame(context.user.position)
    ds = df['stock_code'].map(lambda x: str(x)[2:])
    pos = []
    for _, value in ds.iteritems():
        if value != '000001':
           pos.append(value)
    #print df.drop('000001',axis=1)
    #print df[df['stock_code'] != '000001']
    return pos

def get_xq_profolio_keep_cost_price(context):
    # print xq_user.position
    df = pd.DataFrame(context.user.history)
    df = df[df['status'] == 'success']['rebalancing_histories']
    # print "*****************************"
    # print type(df), df
    _list = []
    for i in df.values:
        _list.append(pd.DataFrame(i))
    histdf = pd.concat(_list)
    histdf = histdf.fillna(0)
    # print histdf.iloc[::-1]
    # print "-------------------"
    print histdf.shape
    tmpdict = {}
    ind = 0
    for _, row in histdf.iloc[::-1].iterrows():  # 获取每行的index、row
        # print type(row), row, type(row['stock_symbol']), str(row['stock_symbol'])[2:]
        stock = str(row['stock_symbol'])[2:]
        if row['volume'] == 0:
            if tmpdict.has_key(stock): del tmpdict[stock]
            continue
        ind += 1
        keep_price = 0
        if tmpdict.has_key(stock):
            keep_price = tmpdict[stock]
        else:
            keep_price = row['prev_price']
        net = row['volume'] - row['prev_volume']
        tmpdict[stock] = (net * row['price'] + row['prev_volume'] * keep_price) / row['volume']
    del tmpdict['000001']  # fix it
    return pd.Series(tmpdict)