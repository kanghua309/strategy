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

Concept_TOPN = 100 #TODO
Concept_Umask = []
Concept_StartNo= 101
'''
def get_sector_size():
    return len(get_sector_class())
'''
