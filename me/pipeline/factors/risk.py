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
from me.pipeline.classifiers.tushare.sector import get_sectors_no
from zipline.pipeline.factors import AverageDollarVolume, CustomFactor
import cvxpy as cvx


MAX_GROSS_LEVERAGE = 1.0
NUM_LONG_POSITIONS = 19 #剔除risk_benchmark
NUM_SHORT_POSITIONS = 0
MAX_BETA_EXPOSURE = 0.20

NUM_ALL_CANDIDATE = NUM_LONG_POSITIONS

MAX_LONG_POSITION_SIZE = 3 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
#MAX_SHORT_POSITION_SIZE = 2*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

MIN_LONG_POSITION_SIZE = 0.3 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
MAX_SECTOR_EXPOSURE = 0.3


def Markowitz(inputs, window_length, mask, trigger_date=None):
    class Markowitz(CustomFactor):
        #params = {'trigger_date': None, }
        def compute(self, today, assets,out,returns,beta):
            print "------------------------------- Markowitz:",today
            if trigger_date != None and today != pd.Timestamp(trigger_date,tz='UTC'):  # 仅仅是最重的预测factor给定时间执行了，其他的各依赖factor还是每次computer调用都执行，也流是每天都执行！ 不理想
                return
            for stock in assets:
                stock = sid(stock).symbol
                print stock

            gamma = cvx.Parameter(sign="positive")
            gamma.value = 1  # gamma is a Parameter that trades off risk and return.
            returns = np.nan_to_num(returns.T)  # time,stock to stock,time
            # [[1 3 2] [3 2 1]] = > [[1 3] [3 2] [2 1]]
            print "return ...\n",  returns
            cov_mat = np.cov(returns)
            print "cov of return:\n", cov_mat
            Sigma = cov_mat
            print "cov values:", Sigma
            candidates_len = len(assets)
            w = cvx.Variable(candidates_len)
            risk = cvx.quad_form(w, Sigma)  # expected_variance => w.T*C*w =  quad_form(w, C)
            print returns
            avg_rets = returns.mean()
            print "avg_rets:", avg_rets
            #target_ret = avg_rets *0.01  #TODO
            target_ret = 0.01  #TODO

            mu = np.array([target_ret] * len(cov_mat))
            expected_return = np.reshape(mu,(-1, 1)).T * w  # w is a vector of stock holdings as fractions of total assets.
            objective = cvx.Maximize(expected_return - gamma * risk)  # Maximize(expected_return - expected_variance)
            # objective = cvx.Maximize(df.pred.as_matrix() * w)  # mini????
            constraints = [cvx.sum_entries(w) == 1.0 * MAX_GROSS_LEVERAGE, w >= 0.0]  # dollar-neutral long/short

            '''
            # constraints.append(cvx.sum_entries(cvx.abs(w)) <= 1)  # leverage constraint
            constraints.extend([w >= MIN_LONG_POSITION_SIZE, w <= MAX_LONG_POSITION_SIZE])  # long exposure
            # risk
            #riskvec = np.nan_to_num(beta)  # TODO
            #print "riskvec:",riskvec
            #print "risk:", beta
            #constraints.extend([riskvec * w <= MAX_BETA_EXPOSURE])  # risk ?
            print "MIN_SHORT_POSITION_SIZE %s, MAX_SHORT_POSITION_SIZE %s,MAX_BETA_EXPOSURE %s" % (
                MIN_LONG_POSITION_SIZE, MAX_LONG_POSITION_SIZE, MAX_BETA_EXPOSURE)
            
            sector_dist = {}
            idx = 0
            class_nos = get_sectors_no(assets)
            print "class_nos:",class_nos
            for classid in class_nos:
                if classid not in sector_dist:
                    _ = []
                    sector_dist[classid] = _
                    sector_dist[classid].append(idx)
                idx += 1
            print"sector size :", len(sector_dist)
            for k, v in sector_dist.iteritems():
                constraints.append(cvx.sum_entries(w[v]) <  (1 + MAX_SECTOR_EXPOSURE) / len(sector_dist))
                constraints.append(cvx.sum_entries(w[v]) >= (1 - MAX_SECTOR_EXPOSURE) / len(sector_dist))
           '''
            prob = cvx.Problem(objective, constraints)
            prob.solve()
            if prob.status != 'optimal':
                print "Optimal failed %s , do nothing" % prob.status
                return None
                # raise SystemExit(-1)
            print np.squeeze(np.asarray(w.value))  # Remo
            out[:] = np.squeeze(np.asarray(w.value)) #每行中的列1

    return Markowitz(inputs = inputs, window_length = window_length, mask = mask)