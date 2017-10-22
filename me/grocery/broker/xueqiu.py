# -*- coding: utf-8 -*-

# import pandas.io.data as web
# from  pandas.stats.api import ols
from datetime import datetime

import easytrader
import pandas as pd

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
class XueqiuLive:
    def __init__(self, user, account, password, portfolio_code, placeholder=None):
        self.exchange = 'xq'
        self.user = user
        self.account = account
        self.password = password
        self.portfolio_code = portfolio_code
        self.placeholder = placeholder

    def login(self):
        self.userbroker = easytrader.use(self.exchange)
        self.userbroker.prepare(user=self.user, account=self.account, password=self.password, portfolio_code=self.portfolio_code)

    def adjust_weight(self,stock,weight):
        self.userbroker.adjust_weight(stock, weight)

    def get_placeholder(self):
        return self.placeholder

    def get_profolio_position(self):
        df = pd.DataFrame(self.userbroker.position)
        ds = df['stock_code'].map(lambda x: str(x)[2:])
        w  =  df['market_value']/df['market_value'].sum()
        s  = pd.Series(w.values,index = ds.values)
        try:
            if self.placeholder != None:
                s = s.drop(self.placeholder)
        except:
            pass
        return s

    def get_profilio_size(self):
        return len(self.userbroker.position) - 1

    def _get_profolio_history(self):
        df = pd.DataFrame(self.userbroker.history)
        df = df[df['status'] == 'success']['rebalancing_histories']
        # print "*****************************"
        # print type(df), df
        _list = []
        for i in df.values:
            _list.append(pd.DataFrame(i))
        if len(_list) == 0:
            return pd.DataFrame()
        histdf = pd.concat(_list)
        histdf = histdf.fillna(0)
        return histdf

    def get_profolio_keep_cost_price(self):
        # print xq_user.position
        histdf = self._get_profolio_history()
        tmpdict = {}
        ind = 0
        for _, row in histdf.iloc[::-1].iterrows():  # 获取每行的index、row
            # print type(row), row, type(row['stock_symbol']), str(row['stock_symbol'])[2:]
            stock = str(row['stock_symbol'])[2:]
            if row['volume'] == 0:
                if tmpdict.has_key(stock): del tmpdict[stock]
                continue
            ind += 1
            if tmpdict.has_key(stock):
                keep_price = tmpdict[stock]
            else:
                keep_price = row['prev_price']
            net = row['volume'] - row['prev_volume']
            tmpdict[stock] = (net * row['price'] + row['prev_volume'] * keep_price) / row['volume']
        s = pd.Series(tmpdict)
        try:
            if self.placeholder != None:
                s = s.drop(self.placeholder)
        except:
            pass
        return s


    def get_profolio_last_trade_day(self):
        histdf = self._get_profolio_history()
        tmpdict = {}
        ind = 0
        for _, row in histdf.iloc[::-1].iterrows():  # 获取每行的index、row
            # print type(row), row, type(row['stock_symbol']), str(row['stock_symbol'])[2:]
            stock = str(row['stock_symbol'])[2:]
            if row['volume'] == 0:
                if tmpdict.has_key(stock): del tmpdict[stock]
                continue
            ind += 1
            if tmpdict.has_key(stock):
                continue
            else:
                lasttime = datetime.utcfromtimestamp(row['created_at']/1000)
            tmpdict[stock] = lasttime

        s = pd.Series(tmpdict)
        try:
            if self.placeholder != None:
                s = s.drop(self.placeholder)
        except:
            pass
        return s

    def get_profolio_info(self):
        histdf = self._get_profolio_history()
        tmpdict = {}
        tmpdict['keep_price'] = {}
        tmpdict['short_time'] = {}
        tmpdict['long_time'] = {}
        print histdf
        for _, row in histdf.iterrows():  # 获取每行的index、row
            #print type(row), row, type(row['stock_symbol']), str(row['stock_symbol'])[2:]
            stock = str(row['stock_symbol'])[2:]
            if row['volume'] == 0:  # 股票被卖
                if stock not in tmpdict['short_time']:  # 只关心最后一次卖出时间
                    tmpdict['short_time'][stock] = datetime.utcfromtimestamp(row['created_at'] / 1000)
            else:
                if stock not in tmpdict['long_time']:   # 只关心最后一次买入
                    tmpdict['long_time'][stock] = datetime.utcfromtimestamp(row['created_at'] / 1000)
            if tmpdict['keep_price'].has_key(stock):
                keep_price = tmpdict['keep_price'][stock]  # 更新最早价格
            else:
                keep_price = row['prev_price']
            net = row['volume'] - row['prev_volume']  # 更新最早价格
            if row['volume'] != 0:
                tmpdict['keep_price'][stock] = (net * row['price'] + row['prev_volume'] * keep_price) / row[
                    'volume']  # 不断更新
        s = pd.DataFrame(tmpdict)
        try:
            if self.placeholder != None:
                s = s.drop(self.placeholder)
        except:
            pass
        return s


