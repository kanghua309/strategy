# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:52:31 2017

@author: kanghua
"""
from zipline.api import (
    sid,
)
from sklearn import preprocessing
import pandas as pd
from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.classifiers import CustomClassifier,Latest
#from me.pipeline.utils import tushare_loader as tu
from me.pipeline.utils.meta import load_tushare_df

Sector_TOPN = 80 #TODO
Sector_Umask = ['服饰','全国地产','区域地产','综合类','商贸代理','陶瓷','啤酒','渔业','其他商业','红黄药酒','水运','白酒']
Sector_StartNo= 101
'''
def get_sector_size():
    return len(get_sector_class())
'''

def get_sector_class(limit_size = Sector_TOPN,umask = Sector_Umask):
    df = load_tushare_df("basic")
    df = df[-df['industry'].isin(umask)]  # 排除给定行
    industryClass={}
    rindustryClass={}
    no = Sector_StartNo
    #for industry,_ in load_tushare_df("industry").groupby('c_name'):
    for industry,_ in df.groupby('industry').industry.value_counts().nlargest(limit_size).iteritems():
        industryClass[industry[0]] = no
        rindustryClass[no] = industry[0]
        no += 1
    return industryClass,rindustryClass

'''
def get_sector(sector_dict=None):
    if sector_dict is None:
        sector_dict = get_sector_class()
    #print("++enter getSector++",len(sector_dict))
    basic=load_tushare_df("basic")
    class Sector(CustomFactor):  
        inputs = [];  
        window_length = 1
        def findSector(self,assets):
            sector_list=[]
            for msid in assets:
                stock = sid(msid).symbol
                try:
                    industry=basic.loc[stock].industry
                    sector_no=sector_dict[industry]
                    sector_list.append(sector_no)
                except:
                    #print "stock %s in industry %s not find in default sector set, set zero" % (stock,industry)
                    sector_list.append(None)
                else:
                    pass
            return sector_list
        def compute(self, today, assets, out, *inputs):
            out[:] = self.findSector(assets)
    return Sector()
'''

def get_sectors_no(mids):
    basic = load_tushare_df("basic")
    _class,_ = get_sector_class()
    no_ = []
    missing_value = 0
    for msid in mids:
        stock = sid(msid).symbol
        try:
            industry = basic.loc[stock].industry
            sector_no = _class[industry]
            no_.append(sector_no)
        except:
            # print "stock %s in industry %s not find in default sector set, set zero" % (stock,industry)
            no_.append(missing_value)
    return no_

def get_sector(sector_dict=None,mask = None,asset_finder = None):
    if sector_dict is None:
        sector_dict,_ = get_sector_class()
    #print("++enter getSector++",len(sector_dict))
    basic=load_tushare_df("basic")
    def _sid(sid):
        return asset_finder.retrieve_asset(sid)
    class Sector(CustomClassifier):  #CustomClassifier 是int , factor 是float
        inputs = []
        window_length = 1
        dtype = np.int64
        missing_value = -1 #似乎不能被返回？？
        #result[isnan(result)] = self.missing_value
        #params = ('universes',)
        def findSector(self,assets):
            sector_list=[]
            for msid in assets:
                if asset_finder != None:
                    stock = _sid(msid).symbol
                else:
                    stock = sid(msid).symbol
                try:
                    industry=basic.loc[stock].industry
                    sector_no=sector_dict[industry]
                    sector_list.append(sector_no)
                except:
                    #print "stock %s in industry %s not find in default sector set, set zero" % (stock,industry)
                    sector_list.append(0)
                else:
                    pass
            return sector_list
        def compute(self, today, assets, out, *inputs):
            rs = self.findSector(assets)
            out[:] = rs
    if mask != None:
        return Sector(mask=mask)
    return Sector()

def get_sector_by_onehot(sector_dict=None,mask = None,asset_finder = None):
    if sector_dict is None:
        sector_dict,_ = get_sector_class()
    basic=load_tushare_df("basic")
    def _sid(sid):
        return asset_finder.retrieve_asset(sid)

    def _onehot_sectors(sector_keys):
        ##- Convert the Sectors column into binary labels
        sector_binarizer = preprocessing.LabelBinarizer()
        strlbls = list(map(str,sector_keys))  # LabelBinarizer didn't like float values, so convert to strings
        sector_binarizer.fit(strlbls)
        sector_labels_bin = sector_binarizer.transform(strlbls)  # this is now 12 binary columns from 1 categorical
        ##- Create a pandas dataFrame from the new binary labels
        #print(sector_labels_bin)
        colNames = []
        for i in range(len(sector_labels_bin[0])):
            colNames.append("S_L_" + strlbls[i])  # TODO
        sLabels = pd.DataFrame(data=sector_labels_bin, index=strlbls, columns=colNames)
        return sLabels

    sector_indict,sector_rindict = get_sector_class() #TODO ORDERDICT????
    #sector_indict_keys = sector_indict.keys()
    #sector_indict_keys.sort()
    sector_indict_keys = sorted(sector_indict)
    onehot_sector = _onehot_sectors(sector_indict_keys)
    #print sector_indict
    #print sector_inddict
    #print onehot_sector
    class OneHotSector(CustomFactor):  #CustomClassifier 是int , factor 是float
        inputs = []
        window_length = 1
        outputs = sector_indict_keys
        def _find_sector(self,asset):
            sector_no = 0
            sector_name = ""
            if asset_finder != None:
                stock = _sid(asset).symbol
            else:
                stock = sid(asset).symbol
            try:
                industry=basic.loc[stock].industry
                sector_no=sector_dict[industry]
                sector_name = industry
            except:
                #print "stock %s in not find in default sector set, set zero" % (stock)
                pass
            else:
                pass
            return sector_no,sector_name

        def compute(self,today, assets, out):
            idx = 0
            for asset in assets:
                sno,sname = self._find_sector(asset)
                if sno != 0:
                    onehots = onehot_sector.loc[sname]
                    #print onehots
                i = 0
                for output in self.outputs:
                    if sno != 0:
                        #print ("++++",idx,output,onehots.values[i])
                        out[idx][output] = int(onehots.values[i])
                    else:
                        out[idx][output] = 0
                    i += 1
                idx += 1
            #print out
    if mask != None:
        return OneHotSector(mask=mask),sector_indict_keys
    else:
        return OneHotSector(),sector_indict_keys



import numpy as np
class RandomUniverse(CustomClassifier):
    inputs = []
    window_length = 1
    dtype = np.int64
    missing_value = 9999
    #params = ('universes',)
    def compute(self, today, assets, out, *inputs):
        out[:] = [0,0,144]
        print("sector:", assets.size, assets,out)
