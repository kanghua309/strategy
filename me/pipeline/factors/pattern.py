# -*- coding: utf-8 -*-

from __future__ import division
from zipline.api import (
    attach_pipeline,
    date_rules,
    time_rules,
    pipeline_output,
    record,
    schedule_function,
    symbol,
    get_datetime,
)
import numpy as np
import pandas as pd

import pandas as pd
import numpy as np
import datetime

from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.signal import argrelextrema
from numpy import linspace
from collections import defaultdict
from zipline.api import (
    symbol,
    sid,
)
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor


def find_max_min(prices):
    prices_ = prices.copy()
    prices_.index = linspace(1., len(prices_), len(prices_))
    #kr = KernelReg([prices_.values], [prices_.index.values], var_type='c', bw=[1.8, 1])
    kr = KernelReg([prices_.values], [prices_.index.values], var_type='c', bw=[2]) # 小了捕捉局部，大了捕捉全局 ！
    # Either a user-specified bandwidth or the method for bandwidth selection.
    # If a string, valid values are ‘cv_ls’ (least-squares cross-validation) and ‘aic’ (AIC Hurvich bandwidth estimation).
    # Default is ‘cv_ls’.
    f = kr.fit([prices_.index.values])

    smooth_prices = pd.Series(data=f[0], index=prices.index)

    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i > 1) and (i < len(prices) - 1):
            price_local_max_dt.append(prices.iloc[i - 2:i + 2].argmax())

    price_local_min_dt = []
    for i in local_min:
        if (i > 1) and (i < len(prices) - 1):
            price_local_min_dt.append(prices.iloc[i - 2:i + 2].argmin())

    prices.name = 'price'
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min.index.name = 'date'
    max_min = max_min.reset_index()
    max_min = max_min[~max_min.date.duplicated()]
    p = prices.reset_index()
    max_min['day_num'] = p[p['index'].isin(max_min.date)].index.values
    max_min = max_min.set_index('day_num').price

    return max_min


def find_patterns(max_min):
    patterns = defaultdict(list)

    for i in range(5, len(max_min) + 1):
        window = max_min.iloc[i - 5:i]

        # pattern must play out in less than 36 days
        if window.index[-1] - window.index[0] > 35:
            continue

        # Using the notation from the paper to avoid mistakes
        e1 = window.iloc[0]
        e2 = window.iloc[1]
        e3 = window.iloc[2]
        e4 = window.iloc[3]
        e5 = window.iloc[4]

        rtop_g1 = np.mean([e1, e3, e5])
        rtop_g2 = np.mean([e2, e4])
        # Head and Shoulders
        if (e1 > e2) and (e3 > e1) and (e3 > e5) and \
                (abs(e1 - e5) <= 0.03 * np.mean([e1, e5])) and \
                (abs(e2 - e4) <= 0.03 * np.mean([e1, e5])):
            patterns['HS'].append((window.index[0], window.index[-1]))

        # Inverse Head and Shoulders
        elif (e1 < e2) and (e3 < e1) and (e3 < e5) and \
                (abs(e1 - e5) <= 0.03 * np.mean([e1, e5])) and \
                (abs(e2 - e4) <= 0.03 * np.mean([e1, e5])):
            patterns['IHS'].append((window.index[0], window.index[-1]))

        # Broadening Top
        elif (e1 > e2) and (e1 < e3) and (e3 < e5) and (e2 > e4):
            patterns['BTOP'].append((window.index[0], window.index[-1]))

        # Broadening Bottom
        elif (e1 < e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
            patterns['BBOT'].append((window.index[0], window.index[-1]))

        # Triangle Top  #越来越低
        elif (e1 > e2) and (e1 > e3) and (e3 > e5) and (e2 < e4):
            patterns['TTOP'].append((window.index[0], window.index[-1]))

        # Triangle Bottom
        elif (e1 < e2) and (e1 < e3) and (e3 < e5) and (e2 > e4):
            patterns['TBOT'].append((window.index[0], window.index[-1]))

        # Rectangle Top
        elif (e1 > e2) and (abs(e1 - rtop_g1) / rtop_g1 < 0.0075) and \
                (abs(e3 - rtop_g1) / rtop_g1 < 0.0075) and (abs(e5 - rtop_g1) / rtop_g1 < 0.0075) and \
                (abs(e2 - rtop_g2) / rtop_g2 < 0.0075) and (abs(e4 - rtop_g2) / rtop_g2 < 0.0075) and \
                (min(e1, e3, e5) > max(e2, e4)):

            patterns['RTOP'].append((window.index[0], window.index[-1]))

        # Rectangle Bottom
        elif (e1 < e2) and (abs(e1 - rtop_g1) / rtop_g1 < 0.0075) and \
                (abs(e3 - rtop_g1) / rtop_g1 < 0.0075) and (abs(e5 - rtop_g1) / rtop_g1 < 0.0075) and \
                (abs(e2 - rtop_g2) / rtop_g2 < 0.0075) and (abs(e4 - rtop_g2) / rtop_g2 < 0.0075) and \
                (max(e1, e3, e5) > min(e2, e4)):
            patterns['RBOT'].append((window.index[0], window.index[-1]))

    return patterns


def _pattern_identification(prices, indentification_lag):
    #print "------_pattern_identification"
    max_min = find_max_min(prices)
    # we are only interested in the last pattern (if multiple patterns are there)
    # and also the last min/max must have happened less than "indentification_lag"
    # days ago otherways it mush have already been identified or it is too late to be usefull
    max_min_last_window = None

    for i in reversed(range(len(max_min))):
        if (prices.index[-1] - max_min.index[i]) == indentification_lag:
            max_min_last_window = max_min.iloc[i - 4:i + 1]
            break

    if max_min_last_window is None:
        return np.nan

    # possibly identify a pattern in the selected window
    patterns = find_patterns(max_min_last_window)
    if len(patterns) != 1:
        return np.nan

    name, start_end_day_nums = patterns.iteritems().next()
    #print(name, start_end_day_nums,max_min_last_window)
    pattern_code = {
        'HS': -2,
        'IHS': 2,
        'BTOP': -1,
        'BBOT': 1,
        'TTOP': -4,
        'TBOT': 4,
        'RTOP': -3,
        'RBOT': 3,
    }

    return pattern_code[name]


class PatternFactor(CustomFactor):
    params = ('indentification_lag',)
    inputs = [USEquityPricing.close]
    window_length = 40

    def compute(self, today, assets, out, close, indentification_lag):
        #print "today=================:",today,len(close),close
        prices = pd.DataFrame(close, columns=assets)
        out[:] = prices.apply(_pattern_identification, args=(indentification_lag,))