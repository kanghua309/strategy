# -*- coding: utf-8 -*-

from strategy import Strategy

class RevertStrategy(Strategy):
    def __init__(self, executor, risk_manager):



        #requests allocation and assigns to an empty dictionary
        self.alloc = dict()
        #records positions associated with strategy and assigns to an empty dictionary
        self.pos = dict()
    def compute_allocation(self,dataframe):
        pass
        raise NotImplementedError()

    def trade(self,dataframe = None):
        raise NotImplementedError()

    def portfolio(self):
        raise NotImplementedError()

    def pipeline_columns_and_mask(self):
        raise NotImplementedError()