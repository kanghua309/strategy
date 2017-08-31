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
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import AverageDollarVolume, CustomFactor
from zipline.pipeline.filters import CustomFilter


from me.pipeline.utils.meta import load_tushare_df

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


def default_china_equity_universe_mask():
    #a_stocks = []
    info = load_tushare_df("basic")
    sme = load_tushare_df("sme")
    gem = load_tushare_df("gem")
    st  = load_tushare_df("st")
    uset = pd.concat([sme, gem, st])
    maskset = info.drop([y for y in uset['code']], axis=0).index  # st,sme,gem 的都不要，稳健型只要主板股票
    #Returns a factor indicating membership (=1) in the given iterable of securities
    #print("==enter IsInSymbolsList==")
    class IsInSecListFactor(CustomFilter):
        inputs = [];
        window_length = 1
        def compute(self, today, asset_ids, out, *inputs):
            #print asset_ids
            #print maskset
            assets  = [sid(id).symbol for id in asset_ids]
            #print "--------------"
            #print pd.Series(assets)
            out[:] = pd.Series(assets).isin(maskset)
            #print out
    return IsInSecListFactor()
