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
from me.pipeline.factors.boost import HurstExp,Beta,SimpleBookToPrice,SimpleMomentum
from me.pipeline.factors.machinelearning import BasicFactorRegress
from me.pipeline.factors.tsfactor import Fundamental
from me.pipeline.filters.universe import make_china_equity_universe, default_china_equity_universe_mask, \
    private_universe_mask

from strategy import Strategy


risk_benchmark = '000001'
class FactorStrategy(Strategy):
    def __init__(self, executor,risk_manager):
        self.executor = executor
        self.risk_manager = risk_manager
        self.portfolio = self.executor.portofolio
        self.portfolio_contain_size = 19
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
        print pipeline_data
        return {},{}

    def trade(self,shorts,longs):
        print "do sell ....."
        self.executor.orders(shorts)
        print "do buy ....."
        self.executor.orders(longs)
        pass

    def portfolio(self):
        raise NotImplementedError()

    def __make_factors(self):
        universe = make_china_equity_universe(
            target_size=3000,
            mask=default_china_equity_universe_mask([risk_benchmark]),
            max_group_weight=0.1,
            smoothing_func=lambda f: f.downsample('month_start'),

        )

        # market cap and book-to-price data gets fed in here
        market_cap = Fundamental().outstanding
        #market_cap.window_safe = True
        #market_cap = Latest([Fundamental().outstanding])


        book_to_price = SimpleBookToPrice()
        #book_to_price.window_safe = True
        # we also get daily returns
        returns = Returns(window_length=2)

        # and momentum as lagged returns (1 month lag)
        momentum = SimpleMomentum()

        # we compute a daily rank of both factors, this is used in the next step,
        # which is computing portfolio membership
        market_cap_rank = market_cap.rank(mask=universe)

        book_to_price_rank = book_to_price.rank(mask=universe)

        momentum_rank = momentum.rank(mask=universe)

        # build Filters representing the top and bottom 1000 stocks by our combined ranking system
        biggest = market_cap_rank.top(100)
        smallest = market_cap_rank.bottom(100)

        highpb = book_to_price_rank.top(100)
        lowpb = book_to_price_rank.bottom(100)

        top = momentum_rank.top(100)
        bottom = momentum_rank.bottom(100)

        universe = universe & ( highpb | lowpb | top | bottom)
        #universe = universe & (biggest | smallest | highpb | lowpb | top | bottom)

        risk_beta = 0.66 * RollingLinearRegressionOfReturns(
            target=symbol(risk_benchmark),  # sid(8554),
            returns_length=6,
            regression_length=21,
            # mask=long_short_screen
            mask=(universe),
        ).beta + 0.33 * 1.0

        all_factors = {
            'market_cap': market_cap,
            'book_to_price': book_to_price.downsample('week_start'),
            #'returns': returns,
            'momentum': momentum,
            'market_beta': risk_beta,
            'biggest': biggest,
            'smallest': smallest,
            'highpb': highpb,
            'lowpb': lowpb,
            'top': top,
            'bottom': bottom,
        }

        return all_factors,universe



    def pipeline_columns_and_mask(self):
        factors,universe = self.__make_factors()
        from collections import OrderedDict
        factors_pipe = OrderedDict()
        # Create returns over last n days.
        factors_pipe['Returns'] = Returns(inputs=[USEquityPricing.open],
                                          mask=universe, window_length=5)
        # Instantiate ranked factors
        idx =0
        for name, f in factors.iteritems():
            #print name
            print "--------------------",name
            f.window_safe = True
            factors_pipe[name] = f


        # Create our ML pipeline factor. The window_length will control how much
        # lookback the passed in data will have.
        predict = BasicFactorRegress(inputs=factors_pipe.values(), window_length=3, mask=universe)
        #print predict
        columns = {
            'predict':predict,
        }
        return columns,universe
