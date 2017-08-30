# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor




class ADV_adj(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 10
    def compute(self, today, assets, out, close, volume):
        print "--------------ADV_adj---------------",today
        close[np.isnan(close)] = 0
        out[:] = np.mean(close * volume, 0)
