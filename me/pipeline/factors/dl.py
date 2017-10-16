# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:42:21 2017

@author: kanghua
"""
import sqlite3

import os
import pandas as pd
from zipline.api import (
    sid,
)
from zipline.pipeline.factors import CustomFactor


def RNNPredict(mask,trigger_date=None,source='History.db'):
    class RNNPredict(CustomFactor):
        inputs = [];
        window_length = 1
        def compute(self, today, assets, out, *inputs):
            if trigger_date != None and today != pd.Timestamp(trigger_date,tz='UTC'):  # 仅仅是最重的预测factor给定时间执行了，其他的各依赖factor还是每次computer调用都执行，也流是每天都执行！ 不理想
                return
            if os.path.splitext(source)[1] == '.db':
                conn = sqlite3.connect(source, check_same_thread=False)  #预测过程由外部程序完成，结果写入到数据库中
                query = "select * from predict where date >= '%s' order by date limit 1 " % str(today)[:19]
                df = pd.read_sql(query, conn)
                df = df.set_index('date')
                conn.close()
            elif os.path.splitext(source)[1] == '.csv':
                 df = pd.read_csv("predict.csv", index_col=0, parse_dates=True)
                 df = df[df.index >= pd.Timestamp(str(today))]
                 print today,df
            else:
                raise ValueError
            new_index = [sid(asset).symbol + "_return" for asset in assets]
            df = df.reindex(columns = new_index)
            out[:] = df.ix[0].values
            print "RNNpredict:", today, out

    return RNNPredict(mask=mask)