# -*- coding: utf-8 -*-

"""
Alpha generator finds edge on market, asks portfolio manager for allocation, and keeps track of
positions held by it.
"""
class Strategy(object):
    def __init__(self, executor = None, risk_manager = None ):
        #requests allocation and assigns to an empty dictionary
        #self.alloc = dict()
        #records positions associated with strategy and assigns to an empty dictionary
        #self.pos = dict()
        pass
    def compute_allocation(self,dataframe):
        pass
        raise NotImplementedError()

    def trade(self,shorts,longs):
        raise NotImplementedError()

    def portfolio(self):
        raise NotImplementedError()

    def pipeline_columns_and_mask(self):
        raise NotImplementedError()
