# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor,Returns, Latest
from me.pipeline.factors.tsfactor import Fundamental
from sklearn import linear_model, decomposition, ensemble, preprocessing, isotonic, metrics
def FactorRegress(columns,inputs,window_length,mask):
    class FactorRegress(CustomFactor):
        init = False
        def compute(self, today, assets, out,*inputs):
            # inputs is a list of factors, for example, assume we have 2 alpha signals, 3 stocks,
            # and a lookback of 2 days. Each element in the inputs list will be data of
            # one signal, so len(inputs) == 2. Then each element will contain a 2-D array
            # of shape [time x stocks]. For example:
            # inputs[0]:
            # [[1, 3, 2], # factor 1 rankings of day t-1 for 3 stocks
            #  [3, 2, 1]] # factor 1 rankings of day t for 3 stocks
            # inputs[1]:
            # [[2, 3, 1], # factor 2 rankings of day t-1 for 3 stocks
            #  [1, 2, 3]] # factor 2 rankings of day t for 3 stocks
            # 和普通factor input 相比，时间要求是多天
            # Usage FactorRegress(inputs=factors_pipe.values(), window_length=window_length + 1, mask=universe)
            print "assets:",assets
            print "inputs:", inputs
            if (not self.init) or (today.weekday == 4):# 周五进行
                # Stack factor rankings
                data = np.hstack(inputs)  # (time, stocks, factors)
            return None
    return FactorRegress(inputs=inputs,window_length=window_length,mask=mask)