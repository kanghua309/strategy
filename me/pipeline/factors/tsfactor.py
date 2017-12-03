# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor
import zipline
from zipline.api import (
    symbol,
    sid,
)
import datetime
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import AverageDollarVolume, CustomFactor


from me.pipeline.utils.meta import load_tushare_df

'''
def MarketCap():
    #print("==enter getMarketCap==")
    info=load_tushare_df("basic")
    class MarketCap(CustomFactor):
        # Shares Outstanding
        inputs = [USEquityPricing.close]
        window_length = 1
        def outstanding(self,assets):
            oslist=[]
            for msid in assets:
                stock = sid(msid).symbol
                try:
                    os = info.ix[stock]['outstanding'] * 1.0e+8
                    oslist.append(os)
                except:
                    oslist.append(0.0)
                else:
                    pass
            return oslist
        def compute(self, today, assets, out, close):
            #print "---------------MarketCap--------------", today
            out[:] =   close[-1] * self.outstanding(assets)
    return MarketCap()
'''
# can be used outside algo scope - give your asset_finder
def Fundamental(mask = None,asset_finder = None):
    def _sid(sid):
        return asset_finder.retrieve_asset(sid)
    columns = ['pe',  # 市盈率
               'outstanding',  # 流通股本(亿)
               'totals',  # 总股本(亿)
               'totalAssets',  # 总资产(万)
               'liquidAssets',  # 流动资产
               'fixedAssets',  # 固定资产
               'reserved',  # 公积金
               'reservedPerShare',  # 每股公积金
               'esp',  # 每股收益
               'bvps',  # 每股净资
               'pb',  # 市净率
               'timeToMarket',  # 上市日期 0：未上市
               'undp',  # 未分利润
               'perundp',  # 每股未分配
               'rev',  # 收入同比(%)
               'profit',  # 利润同比(%)
               'gpr',  # 毛利率(%)
               'npr',  # 净利润率
               'holders',  # 股东人数
               ]
    info=load_tushare_df("basic")
    class Fundamental(CustomFactor):
        outputs = columns
        inputs = [USEquityPricing.close]
        window_length = 1
        window_safe = True
        def handle(self, assets):
            if asset_finder != None:
                stocks = [_sid(msid).symbol for msid in assets]
            else:
                stocks = [sid(msid).symbol for msid in assets]
            #print stocks
            #print info.ix[stocks][columns]
            return info.ix[stocks][columns]
        def compute(self, today, assets, out,close):
            df = self.handle(assets)
            out.pe[:] = df.pe
            out.outstanding[:] =  close[-1] * df.outstanding * 1.0e+8
            out.totals[:] = close[-1] * df.totals * 1.0e+8
            out.totalAssets[:] = df.totalAssets * 1.0e+4
            out.liquidAssets[:] = df.liquidAssets
            out.fixedAssets[:] = df.fixedAssets
            out.reserved[:] =df.reserved
            out.reservedPerShare[:] = df.reservedPerShare
            out.esp[:] = df.esp
            out.bvps[:] = df.bvps
            out.pb[:] = df.pb
            out.timeToMarket[:] = df.timeToMarket
            out.undp[:] = df.undp
            out.perundp[:] = df.perundp
            out.rev[:] = df.rev
            out.profit[:] = df.profit
            out.gpr[:] = df.gpr
            out.npr[:] = df.npr
            out.holders[:] = df.holders
    if mask != None:
        return Fundamental(mask = mask)
    return Fundamental()

'''
def default_china_equity_universe_mask(unmask):
    #a_stocks = []
    info = load_tushare_df("basic")
    sme = load_tushare_df("sme")
    gem = load_tushare_df("gem")
    st  = load_tushare_df("st")
    uset = pd.concat([sme, gem, st])
    maskdf  = info.drop([y for y in uset['code']], axis=0)  # st,sme,gem 的都不要，稳健型只要主板股票
    maskdf = maskdf.drop(unmask,axis=0)
    #Returns a factor indicating membership (=1) in the given iterable of securities
    #print("==enter IsInSymbolsList==")
    class IsInDefaultChinaUniverse(CustomFilter):
        inputs = [];
        window_length = 1
        def compute(self, today, asset_ids, out, *inputs):
            #print asset_ids
            #print maskset
            assets  = [sid(id).symbol for id in asset_ids]
            #print "--------------"
            #print pd.Series(assets)
            out[:] = pd.Series(assets).isin(maskdf.index)
            #print out
    return IsInDefaultChinaUniverse()
'''