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

from me.grocery.strategies.strategy import Strategy

risk_benchmark = '000001'
class DLExampleStrategy(Strategy):
    def __init__(self, executor,risk_manager,predict_time=None):
        self.executor = executor
        self.risk_manager = risk_manager
        self.portfolio = self.executor.portofolio
        self.portfolio_contain_size = 19
        self.predict_time = predict_time
        pass

    def compute_allocation(self,data,pipeline_data):

        xq_profolio_real = self.portfolio[self.portfolio['short_time'].isnull()]
        shorts = {}
        for index, value in xq_profolio_real.iterrows():
            shorts[index] = 0.0

        df = pipeline_data.sort_values(axis=0, by='predict', ascending=False)
        # print "profolio_hold_index:",profolio_hold_index
        weights = self.risk_manager.optimalize(df[:self.portfolio_contain_size],
                                               {'ALPHA': 'predict', 'BETA': 'market_beta', 'SECTOR': 'sector',
                                                "RETURNS": 'returns'})  # 作为参数优化的必备项
        if len(weights) == 0:
            print("Portofolio optimalize failed ,so do nothing")
            return {}, {}
        return shorts, weights.to_dict()

    def trade(self,shorts,longs):
        print ("do sell .....",shorts)
        self.executor.orders(shorts)
        print ("do buy .....", longs)
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
        universe = (universe & last_price) & ~ private_universe
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
        predict  = RNNPredict(universe,source='predict.csv',trigger_date=self.predict_time)  # 进行回顾
        sector = get_sector()

        columns = {
            'predict':predict,
            'market_beta': risk_beta,
            'sector': sector,
        }

        return columns,universe
