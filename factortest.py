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
from zipline.pipeline.factors import AverageDollarVolume, CustomFactor, Latest ,RollingLinearRegressionOfReturns

from me.pipeline.factors.prediction import RNNPredict

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

profolio_size = 10

def make_pipeline():
    beta = 0.66 * RollingLinearRegressionOfReturns(
        target=symbol('000001'),  # sid(8554),
        returns_length=4,
        regression_length=8,
        #mask=long_short_screen
    ).beta + 0.33 * 1.0

    pred = RNNPredict()


    # Build Filters representing the top and bottom 150 stocks by our combined ranking system.
    # We'll use these as our tradeable universe each day.
    rank  = pred.rank()
    short = pred.top(1)
    longs = pred.bottom(1)

    return Pipeline(
        columns={
            'pred':  pred,
            'rank':  rank,
            'beta':  beta,
            'longs': longs,
            'short': short
             #'shorts': test.bottom(2),
        },
    )


def rebalance(context, data):
    # Pipeline data will be a dataframe with boolean columns named 'longs' and
    # 'shorts'.
    print context.sim_params.end_session,type(context.sim_params.end_session),type(get_datetime())
    print("===============================================debug print rebalance TODAY:",get_datetime(),context.sim_params.end_session-get_datetime())
    if (context.sim_params.end_session-get_datetime() > timedelta(days=7) ):
        return
    pipeline_data = context.pipeline_data
    all_assets = pipeline_data.index
    print "all asset"
    print(all_assets)
    print "all pipeline factor"
    print(pipeline_data)
    print "To do profolio rebuild....................."
    df = pipeline_data.sort_values(axis=0, by='pred', ascending=False)
    df.index = [index.symbol for index in df.index]
    print(df)
    import math
    print "value:",math.log(0.95),math.log(1.05)
    xq_profolio = get_xq_profolio_keep_cost_price()
    print "xq_profolio profolio:"
    print xq_profolio
    xq_new_profolio_dict = xq_profolio.to_dict()
    print "--------------check sell-------------------------------------------------------------------------------------"
    for index,value in xq_profolio.iteritems():
        print index,value
        keep_price = xq_profolio[index]
        current_price = data.current(symbol(index), 'price')
        print  index, keep_price, current_price
        if keep_price / current_price > 1.05:
            print "it has down to stop limit, sell it", index
            del xq_new_profolio_dict[index]
        if keep_price / current_price < 0.95:
            print "it has up to expected price , sell it", index
            del xq_new_profolio_dict[index]
        #if df.at[index,"pred"] > math.log(1.05):
        #    print "it predict will down , sell it", index,df.at[index,"pred"]
        #    del xq_new_profolio_dict[index]




    '''
    for index, row in df.iterrows():  # 获取每行的index、row
        #print index,type(row),row
        #for col_name in data.columns:
        #    row[col_name] = exp(row[col_name])  # 把结果返回给data
        if index.symbol not in xq_profolio:
            print index.symbol , "not in xq_profolio"
            continue
        keep_price = xq_profolio[index.symbol]
        current_price = data.current(symbol(index.symbol), 'price')
        print  index.symbol,keep_price,current_price
        if keep_price/current_price > 1.05:
            print "it has down to stop limit, sell it", index
        if keep_price/current_price < 0.95:
            print "it has up to expected price , sell it", index
        if row["pred"] > math.log(1.05):
            print "it predict will down , sell it",index,row["pred"]
            #sell it if in our profolio
            #print type(index.symbol),index.symbol
            #print type(xq_profolio["stock_code"].values),xq_profolio["stock_code"].values
            #if index.symbol in xq_profolio:
            #    print index.symbol," in ",xq_profolio
                #sell_xq(index.symbol)
            #else:
            #    print index.symbol,"not in ",xq_profolio
    '''
    df = df.sort_values(axis=0, by='pred', ascending=True)
    xqdf = df.ix[xq_new_profolio_dict].sort_values(axis=0, by='pred', ascending=False) #给定集合的行
    print xqdf
    print xq_new_profolio_dict
    #xq_profolio = get_xq_profolio()
    print "--------------check buy--------------------------------------------------------------------------------------"
    slotlen = len(xq_new_profolio_dict)
    freeslotlen = profolio_size - slotlen
    print "free slot count:",profolio_size - slotlen

    for index1, row1 in df.iterrows():  # 获取每行的index、row
        if freeslotlen > 0 and not xq_new_profolio_dict.has_key(index1):
           xq_new_profolio_dict[index1] = 0
           continue #先用最高预测吧slot空填了
        #在检查还有没有可替换的东东
        for index2,row2 in xqdf.iterrows():
            if row1["pred"] < math.log(0.95) and not xq_new_profolio_dict.has_key(index1) and xq_new_profolio_dict.has_key(index2):
                print "it predict will up",index1 ,row1["pred"]
                if  abs(row1["pred"] - row2["pred"]) > math.log(0.05):
                    print "instead my profolio:",index2," by",index1
                    print xq_new_profolio_dict[index2]
                    del xq_new_profolio_dict[index2]
                    xq_new_profolio_dict[index1] = 0
                break
















    print "To do cvx optimse profolio weight............"
    import cvxpy as cvx
    print pipeline_data.pred.as_matrix()
    print len(pipeline_data.index)


    print "To do cvx optimse in our new profolio set ..................."
    df = df.ix[xq_new_profolio_dict]

    w = cvx.Variable(len(pipeline_data.index))
    objective = cvx.Maximize(pipeline_data.pred.as_matrix() * w) #mini?

    constraints = [cvx.sum_entries(w) == 1, w > 0 ]  # dollar-neutral long/short
    #constraints.append(cvx.sum_entries(cvx.abs(w)) <= 1)  # leverage constraint
    constraints.extend([w > 0.01,w <= 0.3])  # long exposure
    riskvec = pipeline_data.beta.fillna(1.0).as_matrix()
    MAX_BETA_EXPOSURE = 0.20
    constraints.extend([riskvec * w <= MAX_BETA_EXPOSURE])  # risk

    # filters = [i for i in range(len(africa)) if africa[i] == 1]
    '''
    secMap = {}
    idx = 0
    for i, row in pipeline_data.sector.iteritems():
        print("--", idx, i, row, type(row))
        if row not in secMap:
            x = []
            secMap[row] = x
        secMap[row].append(idx)
        idx += 1
    print(secMap)

    for k, v in secMap.iteritems():
        print(v, 1.0 / len(secMap), len(secMap))
        constraints.append(cvx.sum_entries(w[v]) == 1 / len(secMap))

    # print("risk_factor_exposures.as_matrix().T",pipeline_data.market_beta.fillna(1.0),pipeline_data.market_beta.fillna(1.0).values)
    # constraints.append(pipeline_data.market_beta.fillna(1.0)*w<= MAX_BETA_EXPOSURE)
    '''
    prob = cvx.Problem(objective,
                       constraints)
    prob.solve()
    # if prob.status == 'optimal':
    print prob.status, type(w.value), w.value
    print np.squeeze(np.asarray(w.value)) #Remove single-dimensional entries from the shape of an array

    xq_user = easytrader.use('xq_profolio')
    xq_user.prepare(user='', account='18618280998', password='Threeeyear3#', portfolio_code='ZH1124287')
    #for w in  np.squeeze(np.asarray(w.value)) :
    #     print w,
    df = pd.DataFrame(data = np.transpose((w.value)),columns=pipeline_data.index) #翻转
    '''
    for c in df.columns:
        print c.symbol, ("%.2f" % df.at[0,c])
        weight = df.at[0,c] * 100
        print weight,type(weight)
        try:
            xq_user.adjust_weight(c.symbol,weight)
        except easytrader.webtrader.TradeError,e:
            print e
            pass
        #ret = xq_user.adjust_weight
    '''

def initialize(context):
    attach_pipeline(make_pipeline(), 'my_pipeline')
    schedule_function(rebalance,date_rules.week_start(days_offset=0),half_days = True) #每周一

def handle_data(context, data):
    pass

def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')
    pass


def sell_xq_all():
    xq_user = easytrader.use('xq')
    xq_user.prepare(user='', account='18618280998', password='Threeeyear3#', portfolio_code='ZH1135253')
    for pos in xq_user.position:
        print pos
        print pos['stock_code'][2:]
        if (pos['stock_code'][2:] != '000001'):
            xq_user.adjust_weight(pos['stock_code'][2:], 0.0)

def sell_xq(stock):
    xq_user = easytrader.use('xq')
    xq_user.prepare(user='', account='18618280998', password='Threeeyear3#', portfolio_code='ZH1135253')
    xq_user.adjust_weight(stock, 0.0)


def get_xq_profolio_keep_cost_price():
    xq_user = easytrader.use('xq')
    xq_user.prepare(user='', account='18618280998', password='Threeeyear3#', portfolio_code='ZH1135253')
    #print xq_user.position
    df = pd.DataFrame(xq_user.history)
    df = df[df['status'] == 'success']['rebalancing_histories']
    #print "*****************************"
    #print type(df), df
    _list = []
    for i in df.values:
        _list.append(pd.DataFrame(i))
    histdf = pd.concat(_list)
    histdf = histdf.fillna(0)
    #print histdf.iloc[::-1]
    #print "-------------------"
    print histdf.shape
    tmpdict = {}
    ind = 0
    for _, row in histdf.iloc[::-1].iterrows():  # 获取每行的index、row
        #print type(row), row, type(row['stock_symbol']), str(row['stock_symbol'])[2:]
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
    del tmpdict['000001'] #fix it
    return pd.Series(tmpdict)