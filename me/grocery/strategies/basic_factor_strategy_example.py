# -*- coding: utf-8 -*-

from zipline.pipeline.data import USEquityPricing
from zipline.api import (
    symbol,
    get_datetime,
    order_percent,
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
class BasicFactorStrategy(Strategy):
    def __init__(self , risk_manager):
        self.risk_manager = risk_manager
        self.portfolio_contain_size = 20
        self.stocks = {}
        pass

    def compute_allocation(self,data,pipeline_data):
        print (self.stocks)
        shorts = {}
        for stock in self.stocks:
            print (stock)
            shorts[stock] = 0.0

        df = pipeline_data.sort_values(axis=0, by='predict', ascending=False)
        # print "profolio_hold_index:",profolio_hold_index
        weights = self.risk_manager.optimalize(df[:self.portfolio_contain_size],
                                               {'ALPHA': 'predict', 'BETA': 'market_beta', 'SECTOR': 'sector',
                                                "RETURNS": 'returns'})  # 作为参数优化的必备项
        if len(weights) == 0:
            print("Portofolio optimalize failed ,so do nothing")
            return {},{}
        return shorts,weights.to_dict()

    def trade(self,shorts,longs):
        print("do sell .....",shorts)
        for stock in shorts:
            order_percent(symbol(stock),shorts[stock])
            del self.stocks[stock]
        print("do buy .....",longs)
        for stock in longs:
            order_percent(symbol(stock), longs[stock])
            self.stocks[stock] = True
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

        last_price = USEquityPricing.close.latest >= 1.0  # 大于1元
        universe = universe & last_price

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
        for name, f in factors.items():
            f.window_safe = True
            factors_pipe[name] = f.rank(mask=universe) #rank 使用相对顺序，而不是绝对值，避免自相似性

        predict = BasicFactorRegress(inputs=factors_pipe.values(), window_length=42, mask=universe) #进行预测，5天后价格
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
