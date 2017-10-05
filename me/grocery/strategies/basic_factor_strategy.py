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

from .strategy import Strategy


risk_benchmark = '000001'
class FactorStrategy(Strategy):
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
                print ("%s has down to stop limit, sell it - for %s,%s " % (index, keep_price, current_price))
                stop_dict[index] = 0.0
            if keep_price / current_price < 0.90:
                print ("%s has up to expected price , sell it - for %s,%s" % (index, keep_price, current_price))
                stop_dict[index] = 0.0

        return stop_dict

    def __check_expired_limit(self,data):
        profolio = self.portfolio[self.portfolio['short_time'].isnull()].long_time
        stop_dict = {}
        for index, value in profolio.iteritems():
            lastdt = profolio[index]
            # print "Rebalance - index, keep_price, current_price"
            if datetime.now() - lastdt > timedelta(weeks=2):
                print ("%s has expired , sell it - for %s,%s" % (index, datetime.now(), lastdt))
                stop_dict[index] = 0.0
        return stop_dict

    def compute_allocation(self,data,pipeline_data):
        # print pipeline_data.loc['000018']
        # context.xueqiuLive.login()
        print ("Rebalance - Current xq profolio")
        # print len(self.portfolio), self.portfolio

        xq_profolio_real = self.portfolio[self.portfolio['short_time'].isnull()]
        remove_dict = self.__check_stop_limit(data)
        print ("Rebalance - remove_stock for stop:", remove_dict)
        _remove = self.__check_expired_limit(data)
        remove_dict.update(_remove)  # TODO
        print ("Rebalance - remove_stock for expire:", remove_dict)
        profolio_hold_index = xq_profolio_real.index.difference(remove_dict)

        # print "profolio_hold_index:",profolio_hold_index
        print ("Rebalance - Profolio_hold_index now:", profolio_hold_index)
        profolio_hold = pipeline_data.loc[profolio_hold_index]
        weights = self.risk_manager.optimalize(profolio_hold,
                                               {'ALPHA': 'predict', 'BETA': 'market_beta', 'SECTOR': 'sector',
                                                "RETURNS": 'returns'})  # 作为参数优化的必备项
        return remove_dict, weights

    def trade(self,shorts,longs):
        print ("do sell .....")
        self.executor.orders(shorts)
        print ("do buy .....")
        self.executor.orders(longs)
        pass

    def portfolio(self):
        raise NotImplementedError()

    def __make_factors(self):
        universe = make_china_equity_universe(
            target_size=3000,
            mask=default_china_equity_universe_mask([risk_benchmark]),
            max_group_weight=0.01,
            smoothing_func=lambda f: f.downsample('month_start'),

        )
        private_universe = private_universe_mask(self.portfolio.index)  # 把当前组合的stock 包含在universe中

        last_price = USEquityPricing.close.latest >= 1.0  # 大于1元
        universe = universe & last_price | private_universe

        # market cap and book-to-price data gets fed in here
        outstanding = Fundamental().outstanding
        outstanding.window_safe = True
        market_cap = Latest([outstanding])

        book_to_price = SimpleBookToPrice()

        momentum = SimpleMomentum(mask=universe)


        alpha5 = Alpha5(mask=universe)
        alpha8 = Alpha8(mask=universe)
        alpha9 = Alpha9(mask=universe)



        all_factors = {
            'market_cap': market_cap.downsample('month_start'),
            'book_to_price': book_to_price.downsample('month_start'),
            'momentum': momentum,
            'alpha5': alpha5,
            'alpha8': alpha8,
            'alpha9': alpha9,
        }

        return all_factors,universe


    def pipeline_columns_and_mask(self):
        factors,universe = self.__make_factors()
        from collections import OrderedDict
        factors_pipe = OrderedDict()
        # Create returns over last n days.
        factors_pipe['Returns'] = Returns(inputs=[USEquityPricing.close],
                                          mask=universe, window_length=5)
        # Instantiate ranked factors
        for name, f in factors.iteritems():
            f.window_safe = True
            factors_pipe[name] = f.rank(mask=universe) #rank 使用相对顺序，而不是绝对值，避免自相似性

        predict = BasicFactorRegress(inputs=factors_pipe.values(), window_length=42, mask=universe, trigger_date=self.predict_time) #进行回顾
        risk_beta = 0.66 * RollingLinearRegressionOfReturns(
            target=symbol(risk_benchmark),  # sid(8554),
            returns_length=6,
            regression_length=21,
            # mask=long_short_screen
            mask=(universe),
        ).beta + 0.33 * 1.0
        sector = get_sector()

        columns = {
            'market_beta':risk_beta,
            'sector':sector,
            'predict':predict,
        }
        return columns,universe
