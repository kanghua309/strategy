# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:42:21 2017

@author: kanghua
"""
import numpy as np
import pandas as pd

from zipline.api import (
    symbol,
    sid,
)
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor
import sqlite3

def RNNPredict(mask,trigger_date=None):
    class RNNPredict(CustomFactor):
        inputs = [];
        window_length = 1
        def compute(self, today, assets, out, *inputs):
            #print "=====",type(today),today,today.weekday()
            if trigger_date != None and today != pd.Timestamp(trigger_date,tz='UTC'):  # 仅仅是最重的预测factor给定时间执行了，其他的各依赖factor还是每次computer调用都执行，也流是每天都执行！ 不理想
                return
                #print "----",today
            data = today
            #print "today",type(today),today
            #print "asset",type(assets),assets
            #print "out",type(out),out
            #print "input",type(inputs),inputs
            conn = sqlite3.connect('History.db', check_same_thread=False)  #预测过程由外部程序完成，结果写入到数据库中
            query = "select * from predict where date >= '%s' order by date limit 1 " % str(data)[:19]
            #query = "select * from predict where date <= '%s' order by date limit 1 " % str(data)[:19]

            #print query
            df = pd.read_sql(query, conn)
            df = df.set_index('date')
            #print df.head(10)
            #print "-----------read from sql---------"
            new_index = [sid(asset).symbol + "return" for asset in assets]
            #print "new_index:",new_index
            df = df.reindex(columns = new_index)
            #print df.head(10)
            #print type(df.columns.values),
            #print df[-1:].values
            #print df.ix[0].values
            out[:] = df.ix[0].values
            conn.close()
            #print "RNNpredict:",out
    return RNNPredict(mask=mask)