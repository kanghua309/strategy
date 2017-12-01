# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:52:31 2017

@author: kanghua
"""
from zipline.api import (
    sid,
)
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
    no = Sector_StartNo
    #for industry,_ in load_tushare_df("industry").groupby('c_name'):
    for industry,_ in df.groupby('industry').industry.value_counts().nlargest(limit_size).iteritems():
        industryClass[industry[0]] = no
        no = no +1
    return industryClass

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
    _class = get_sector_class()
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
        sector_dict = get_sector_class()
    #print("++enter getSector++",len(sector_dict))
    basic=load_tushare_df("basic")
    #print(asset_finder)
    def _sid(sid):
        return asset_finder.retrieve_asset(sid)
    class Sector(CustomClassifier):  #CustomClassifier 是int , factor 是float
        inputs = []
        window_length = 1
        dtype = np.int64
        missing_value = 999 #似乎不能被返回？？
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
            #out[:] = self.findSector(assets)
            rs = self.findSector(assets)
            #print("sector:",assets.size,assets,rs)
            #out[:] = [0, 0, 144]
            out[:] = rs
    return Sector(mask=mask)


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
