# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:52:31 2017

@author: kanghua
"""
from zipline.api import (
    sid,
)
from zipline.pipeline.factors import CustomFactor
#from me.pipeline.utils import tushare_loader as tu
from me.pipeline.utils.meta import load_tushare_df

Sector_TOPN = 80 #TODO

def get_sector_size():
    return len(get_sector_class())

def get_sector_class():
    industryClass={}
    no = 101
    #for industry,_ in load_tushare_df("industry").groupby('c_name'):
    for industry,_ in load_tushare_df("basic").groupby('industry').industry.value_counts().nlargest(Sector_TOPN).iteritems():
        industryClass[industry[0]] = no
        no = no +1
    return industryClass


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


