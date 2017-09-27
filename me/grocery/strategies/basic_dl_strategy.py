# -*- coding: utf-8 -*-

from zipline.pipeline.data import USEquityPricing
from zipline.api import (
    symbol,
    get_datetime,
)

import numpy as np
import itertools
import statsmodels.api as sm
from statsmodels import regression,stats
import scipy

from datetime import timedelta, datetime
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import RollingLinearRegressionOfReturns,Latest,Returns
from me.pipeline.classifiers.tushare.sector import get_sector
from me.pipeline.factors.boost import HurstExp,Slope,SimpleBookToPrice,SimpleMomentum
from me.pipeline.factors.alpha101 import Alpha5,Alpha8,Alpha9
from me.pipeline.factors.ml import BasicFactorRegress
from me.pipeline.factors.tsfactor import Fundamental
from me.pipeline.filters.universe import make_china_equity_universe, default_china_equity_universe_mask, \
    private_universe_mask
from me.pipeline.factors.risk import Markowitz
from me.pipeline.factors.dl import RNNPredict

from strategy import Strategy


risk_benchmark = '000001'
class DLStrategy(Strategy):
    def __init__(self, executor,risk_manager,predict_time=None):
        self.executor = executor
        self.risk_manager = risk_manager
        self.portfolio = self.executor.portofolio
        self.portfolio_contain_size = 19
        self.predict_time = predict_time
        pass

    def __check_stop_limit(self,data):
        profolio = self.portfolio[self.portfolio['short_time'].isnull()].keep_price
        stop_dict = {}
        for index, value in profolio.iteritems():
            keep_price = profolio[index]
            current_price = data.current(symbol(index), 'price')
            # print "Rebalance - index, keep_price, current_price"
            if keep_price / current_price > 1.10:
                print "%s has down to stop limit, sell it - for %s,%s " % (index, keep_price, current_price)
                stop_dict[index] = 0.0
            if keep_price / current_price < 0.90:
                print "%s has up to expected price , sell it - for %s,%s" % (index, keep_price, current_price)
                stop_dict[index] = 0.0

        return stop_dict

    def __check_expired_limit(self,data):
        profolio = self.portfolio[self.portfolio['short_time'].isnull()].long_time
        stop_dict = {}
        for index, value in profolio.iteritems():
            lastdt = profolio[index]
            # print "Rebalance - index, keep_price, current_price"
            if datetime.now() - lastdt > timedelta(weeks=2):
                print "%s has expired , sell it - for %s,%s" % (index, datetime.now(), lastdt)
                stop_dict[index] = 0.0
        return stop_dict

    def compute_allocation(self,data,pipeline_data):
        # print pipeline_data.loc['000018']
        # context.xueqiuLive.login()
        weights = pipeline_data.weights
        return {},weights

    def trade(self,shorts,longs):
        print "do sell ....."
        self.executor.orders(shorts)
        print "do buy ....."
        self.executor.orders(longs)
        pass

    def portfolio(self):
        raise NotImplementedError()


    def pipeline_columns_and_mask(self):
        universe = make_china_equity_universe(
            target_size=3000,
            mask=default_china_equity_universe_mask([risk_benchmark]),
            max_group_weight=0.01,
            smoothing_func=lambda f: f.downsample('month_start'),

        )
        private_universe = private_universe_mask(self.portfolio.index)  # 把当前组合的stock 包含在universe中
        last_price = USEquityPricing.close.latest >= 1.0  # 大于1元
        universe = (universe & last_price) | private_universe
        # print "universe:",universe
        # Instantiate ranked factors
        returns = Returns(inputs=[USEquityPricing.close],mask=universe,window_length=2)
        risk_beta = 0.66 * RollingLinearRegressionOfReturns(
           target=symbol(risk_benchmark),  # sid(8554),
           returns_length=5,
           regression_length=21,
           # mask=long_short_screen
           mask=(universe),
        ).beta + 0.33 * 1.0
        returns.window_safe = True
        risk_beta.window_safe = True

        predict = RNNPredict(universe, trigger_date=self.predict_time)  # 进行回顾
        universe = predict.top(20)
        weights  = Markowitz(inputs=[returns,risk_beta],window_length=4, mask=universe,trigger_date=self.predict_time) #进行回顾

        columns = {
            'predict':predict,
            'weights':weights,
        }

        return columns,universe
