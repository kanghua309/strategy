# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:42:21 2017

@author: kanghua
"""
import numpy as np
import pandas as pd

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor, Returns, Latest ,RSI
from me.pipeline.factors.tsfactor import Fundamental
from me.pipeline.classifiers.tushare.sector import get_sector,RandomUniverse,get_sector_class,get_sector_by_onehot
from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.classifiers import CustomClassifier,Latest
import talib

class ILLIQ(CustomFactor):
    inputs = [USEquityPricing.close,USEquityPricing.volume]
    window_length = int(252)
    def compute(self, today, assets, out, close,volume):
        window_length = len(close)
        #print "window length",window_length
        _rets = np.abs(pd.DataFrame(close, columns=assets).pct_change()[1:])
        #print(_rets.head(10))
        _vols  = pd.DataFrame(volume, columns= assets)[1:]
        #print(_vols.head(10))
        #print "--------------------------1"
        #print(_rets/_vols)
        #print "--------------------------2"
        #print pd.rolling_mean(_rets/_vols, window=window_length-1)
        #print (_rets/_vols).mean(),type((_rets/_vols).mean())
        out[:] =(_rets/_vols).mean().values

class MeanReturn(CustomFactor):
    inputs = [Returns(window_length=2)]
    def compute(self, today, assets, out, returns):
        out[:] = np.nanmean(returns, axis=0)


class OneHotSector(CustomFactor):
    inputs = [RandomUniverse()]
    #window_length = 1
    window_safe = True
    #dtype = np.int64
    missing_value = 0
    #outputs = get_sector_class()[0]
    #outputs,_ = get_sector_class()
    outputs = ['highs', 'lows']
    xssss,yssss = get_sector_class()
    def oneHot_sectors(self,sector_keys):
        ##- Convert the Sectors column into binary labels
        from sklearn import preprocessing
        import pandas as pd
        sector_binarizer = preprocessing.LabelBinarizer()
        strlbls = map(str, sector_keys)  # LabelBinarizer didn't like float values, so convert to strings
        print "strlbls",type(strlbls),strlbls
        sector_binarizer.fit(strlbls)
        sector_labels_bin = sector_binarizer.transform(strlbls)  # this is now 12 binary columns from 1 categorical

        ##- Create a pandas dataFrame from the new binary labels
        print(sector_labels_bin)
        colNames = []
        for i in range(len(sector_labels_bin[0])):
            colNames.append("S_Label_" + strlbls[i] + str(i)) #TODO
        sLabels = pd.DataFrame(data=sector_labels_bin, index=sector_keys, columns=colNames)
        return sLabels

    def compute(self, today, assets, out,input):
        print "===",self.outputs
        print "+++",type(input),np.shape(input),input
        print "###",assets
        print "&&&",type(out),np.shape(out),out
        print type(get_sector_class()),type(self.outputs),type(self.yssss)
        print self.yssss
        rs = self.oneHot_sectors(self.outputs)
        print rs
        print rs.index
        i = 0
        for no in input[0,:].tolist():
            print "-----------no",no
            try:
                print self.yssss[no]
                #print rs.loc(self.yssss[no])
                print rs.iloc[-1]
                print type(rs.iloc[-1].values),rs.iloc[-1].values
                #print("+++++++++++++++++++++++++++++++1")
                #print rs.loc('汽车整车')
                #print("+++++++++++++++++++++++++++++++2")
                #print rs.loc(u'种植业')
                print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++"
                print(out.dtype.names)
                print out[i]

                # print(type(out[i]),out[i],rs.iloc[-1].values)
                # out[i] =tuple(rs.iloc[-1].values)
                # print("===========================================")
                # print out[i]
                j = 0
                for x in self.outputs:
                    out[i][x] = int(rs.iloc[-1].values[j])
                    j += 1

                i += 1
                print("out ................................. 0")
                #print out[i, :]
            except Exception as e:
                print e
                pass
        print "out ------------------------------------------------------------------------ "
        print out



        #dic = get_sector_class()


        #print rs[x]




# class MOM(CustomFactor):
#     # this class generates the MACD as a Percentage
#     inputs = [USEquityPricing.close]
#     window_length = int(252)
#
#     def columnwise_anynan(self,array2d):
#         return np.isnan(array2d).any(axis=0)
#     def compute(self, today, assets, out, close):
#             window_length= len(close)
#             print window_length,np.shape(close)
#             anynan = self.columnwise_anynan(close)
#             for col_ix, have_nans in enumerate(anynan):
#                 if have_nans:
#                     out[col_ix] = np.nan
#                     continue
#                 print(window_length,close[:, col_ix])
#                 mom = talib.MOM(close[:, col_ix],timeperiod=window_length-1)
#
#                 out[col_ix] = mom[-1]



import click
import numpy as np
import pandas as pd
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Latest
from zipline.pipeline.factors import Returns,AverageDollarVolume
from zipline.utils.cli import Date

from me.helper.research_env import Research
from me.pipeline.factors.tsfactor import Fundamental
from me.pipeline.filters.universe import make_china_equity_universe, default_china_equity_universe_mask, \
    private_universe_mask
from zipline.utils.cli import Date, Timestamp


start = '2015-9-1'  # 必须在国内交易日
end   = '2015-9-10'  # 必须在国内交易日

c,_ = get_sector_class()
ONEHOTCLASS = tuple(c)

def make_pipeline(asset_finder):
    # h2o = USEquityPricing.high.latest / USEquityPricing.open.latest
    # l2o = USEquityPricing.low.latest / USEquityPricing.open.latest
    # c2o = USEquityPricing.close.latest / USEquityPricing.open.latest
    # h2c = USEquityPricing.high.latest / USEquityPricing.close.latest
    # l2c = USEquityPricing.low.latest / USEquityPricing.close.latest
    # h2l = USEquityPricing.high.latest / USEquityPricing.low.latest
    #
    # vol = USEquityPricing.volume.latest
    # outstanding = Fundamental(asset_finder).outstanding
    # outstanding.window_safe = True
    # turnover_rate = vol / Latest([outstanding])
    # returns = Returns(inputs=[USEquityPricing.close], window_length=5)  # 预测一周数据
    #private_universe = private_universe_mask(['000001','000002','000005','000004','000006','000007''000009'],asset_finder=asset_finder)
    private_universe = private_universe_mask(['000001','000002','000005'],asset_finder=asset_finder)

    #illiq = ILLIQ(window_length=20)

    # illiq = ILLIQ(window_length=20,mask = private_universe)
    # ep = 1/Fundamental(asset_finder).pe
    # bp = 1/Fundamental(asset_finder).pb
    # bvps = Fundamental(asset_finder).bvps
    # rev20 = Returns(inputs=[USEquityPricing.close], window_length=20)
    # vol20 = AverageDollarVolume(window_length=20)
    rsi = RSI(window_length=20,mask = private_universe)
    #market = Fundamental(asset_finder).outstanding
    sector = get_sector(asset_finder=asset_finder,mask=private_universe)
    random = RandomUniverse(mask = private_universe)
    #returns = Returns(window_length=50)
    #mr = MeanReturn(inputs=[returns], window_length=252, mask=private_universe)
    #random.window_safe = True
    ONEHOTCLASS,sector_indict_keys = get_sector_by_onehot(asset_finder=asset_finder,mask=private_universe)

    #ONEHOTCLASS = OneHotSector(inputs=[random],window_length=1, mask=private_universe)

    pipe_columns = {
        # 'h2o': h2o.log1p().zscore(),
        # 'l2o': l2o.log1p().zscore(),
        # 'c2o': c2o.log1p().zscore(),
        # 'h2c': h2c.log1p().zscore(),
        # 'l2c': l2c.log1p().zscore(),
        # 'h2l': h2l.log1p().zscore(),
        # 'vol': vol.zscore(),
        # 'turnover_rate': turnover_rate.log1p().zscore(),
        # 'return': returns.log1p(),
        # 'ILLIQ':illiq,
        # 'ep':ep,
        # 'vol20':vol20,
        # #'rsi':rsi.zscore(groupby = sector,mask=rsi.percentile_between(1, 99)),
        'rsi0': rsi,
        'rsi1': rsi.zscore(),
        'rsi2': rsi.zscore(groupby=sector),
        # 'rsi3': vol20.demean(groupby=sector),
        # 'market_rank':market.quantiles(100),
        'sector':sector,
        #'ohs':ohs,
    }
    # pipe_screen = (low_returns | high_returns)
    pipe = Pipeline(columns=pipe_columns,
           screen=private_universe,
           )

    i = 0 #TODO
    for c in ONEHOTCLASS:
        #print "xxxx",sector_indict_keys[i]
        pipe.add(c,sector_indict_keys[i])
        i += 1

    return pipe




pd.set_option('display.width', 800)
research = Research()
#print(research.get_engine()._finder)
my_pipe = make_pipeline(research.get_engine()._finder)
result = research.run_pipeline(my_pipe,
                               Date(tz='utc', as_timestamp=True).parser(start),
                               Date(tz='utc', as_timestamp=True).parser(end))
print result
