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
import datetime
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import AverageDollarVolume, CustomFactor

class FixTime(CustomFactor):
    params = {'trigger_date':None,}
    window_length = 1
    def compute(self, today, assets, out,factor,trigger_date):
        print "trigger_date:", trigger_date, today
        if trigger_date != None:
            try:
                if today != pd.Timestamp(trigger_date,tz='UTC'):
                    out[:] = None
                else:
                    print "------------------------------------================= do real factor?:"
                    print factor.T  #  factor 一天一行 .T一个股票一行

                    print "-----------------------------------",type(factor[-1])
                    out[:] = factor[-1]
            except:
                out[:] = None
        else:
            out[:] = None
