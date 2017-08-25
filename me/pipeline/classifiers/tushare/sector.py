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


SECTOR_NAMES = {
101:'Transport',
102:'Instrument',
103:'Media Entertainment',
104:'Water Supply',
105:'Highway Bridge',
106:'Other Industries',
107:'Animal Husbandry And Fishery',
108:'Pesticide Fertilizer',
109:'Chemical Industry',
110:'Chemical Fiber Industry',
111:'Medical Devices',
112:'Printing Packaging',
113:'Power Generation Equipment',
114:'Business Department',
115:'Plastics',
116:'Furniture Industry',
117:'Appliance Industry',
118:'Building Materials',
119:'Development Zone',
120:'Real Estate',
121:'Motorcycle',
122:'Nonferrous Metals',
123:'Clothing Footwear',
124:'Machinery Industry',
125:'New Shares',
126:'Cement Industry',
127:'Car Manufacturing',
128:'Coal Industry',
129:'Material Trade',
130:'Environmental Protection Industry',
131:'Glass Industry',
132:'Biopharmaceuticals',
133:'Power Industry',
134:'Electronic Industry',
135:'Electronic Information',
136:'Electronic Devices',
137:'Oil Industry',
138:'Textile Machinery',
139:'Textile Industry',
140:'Integrated Industry',
141:'Ship Manufacturing',
142:'Paper industry',
143:'Hotel Tour',
144:'Wine Industry',
145:'Financial Industry',
146:'Steel Industry',
147:'Ceramics Industry',
148:'Aircraft Manufacturing',
149:'Food Industry'
}

def getSectorSize():
    return len(SECTOR_NAMES)

def industryClassied():
    g_inds={}
    no = 101
    for k1, group in load_tushare_df("industry").groupby('c_name'):
        #print k1
        #print group
        g_inds[k1] = no
        no = no +1
    return g_inds;
 

def getSector(ind_dict=None):
    if ind_dict is None:
       ind_dict = industryClassied()

    print("++enter getSector++",len(ind_dict))
    tmp=load_tushare_df("industry")
    class Sector(CustomFactor):  
        inputs = [];  
        window_length = 1
        def findSector(self,assets):
            sector_list=[]
            for msid in assets:
                stock = sid(msid).symbol
                try:
                    ind=tmp[tmp['code']==stock]['c_name'].values[0]
                    #print ind
                    ino=ind_dict[ind]
                    sector_list.append(ino)
                except:
                    #print "not find"
                    sector_list.append(0)
                else:
                    pass
            return sector_list
        def compute(self, today, assets, out, *inputs):
            out[:] = self.findSector(assets)
    return Sector()


