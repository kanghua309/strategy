# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from riskmanager import RiskManager
import cvxpy as cvx

MAX_GROSS_LEVERAGE = 1.0
NUM_LONG_POSITIONS = 19 #剔除risk_benchmark
NUM_SHORT_POSITIONS = 0
MAX_BETA_EXPOSURE = 0.20

NUM_ALL_CANDIDATE = NUM_LONG_POSITIONS



MAX_LONG_POSITION_SIZE = 5 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
#MAX_SHORT_POSITION_SIZE = 2*1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)

MIN_LONG_POSITION_SIZE = 0.1 * 1.0/(NUM_LONG_POSITIONS + NUM_SHORT_POSITIONS)
MAX_SECTOR_EXPOSURE = 0.50



class BasicHedgeRiskManager(RiskManager):

    def __init__(self):
        pass

    #@staticmethod #?
    def optimalize(self,candidates,factors):
        print candidates.index
        print "candidates len:", len(candidates),candidates
        print "factors:",factors


        candidates_len = candidates.index
        w = cvx.Variable(len(candidates_len))
        # objective = cvx.Maximize(df.pred.as_matrix() * w)  # mini????
        objective = cvx.Maximize(candidates[factors.BETA].as_matrix() * w)           #FIX IT
        constraints = [cvx.sum_entries(w) == 1.0 * MAX_GROSS_LEVERAGE, w >= 0.0]  # dollar-neutral long/short
        # constraints.append(cvx.sum_entries(cvx.abs(w)) <= 1)  # leverage constraint
        constraints.extend([w >= MIN_LONG_POSITION_SIZE, w <= MAX_LONG_POSITION_SIZE])  # long exposure
        riskvec = candidates[factors['BETA']].fillna(1.0).as_matrix()                        # TODO
        constraints.extend([riskvec * w <= MAX_BETA_EXPOSURE])                    # risk
        print "MIN_SHORT_POSITION_SIZE %s, MAX_SHORT_POSITION_SIZE %s,MAX_BETA_EXPOSURE %s" % (MIN_LONG_POSITION_SIZE, MAX_LONG_POSITION_SIZE, MAX_BETA_EXPOSURE)
        # 版块对冲当前，因为股票组合小，不合适
        sector_dist = {}
        idx = 0
        for equite, classid in candidates[factors['SECCTOR']].iteritems():
            if classid not in sector_dist:
                _ = []
                sector_dist[classid] = _
                sector_dist[classid].append(idx)
            idx += 1
        print"sector size :", len(sector_dist)
        for k, v in sector_dist.iteritems():
            constraints.append(cvx.sum_entries(w[v]) <  (1 + MAX_SECTOR_EXPOSURE) / len(sector_dist))
            constraints.append(cvx.sum_entries(w[v]) >= (1 - MAX_SECTOR_EXPOSURE) / len(sector_dist))
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        if prob.status != 'optimal':
            print "Optimal failed %s , do nothing" % prob.status
            return pd.Series()
            # raise SystemExit(-1)
        print np.squeeze(np.asarray(w.value))  # Remove single-dimensional entries from the shape of an array
        return pd.Series(data=np.squeeze(np.asarray(w.value)), index=candidates.index)



