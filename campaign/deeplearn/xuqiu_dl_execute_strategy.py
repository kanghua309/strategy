# -*- coding: utf-8 -*-

import os
import pandas as pd
from zipline.api import (
    symbol,
    sid,
)
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor
from zipline.pipeline.factors import RollingLinearRegressionOfReturns, Returns

from me.grocery.strategies.strategy import Strategy
from me.pipeline.classifiers.tushare.sector import get_sector

RISK_BENCHMARK = '000001'  # 平安作为对比股

def RNNPredict(mask, trigger_date=None, source='predict.csv'):
    class RNNPredict(CustomFactor):
        inputs = [];
        window_length = 1

        def compute(self, today, assets, out, *inputs):
            if trigger_date != None and today != pd.Timestamp(trigger_date, tz='UTC'):
                return
            if os.path.splitext(source)[1] == '.csv':
                df = pd.read_csv(source, index_col=0, parse_dates=True)
                # df = df[df.index >= pd.Timestamp(str(today))]
                print today, df
            else:
                raise ValueError
            new_index = [sid(asset).symbol + "_return" for asset in assets]
            df = df.reindex(columns=new_index)
            out[:] = df.ix[0].values
            print "RNNpredict:", today, out

    return RNNPredict(mask=mask)

class DLExampleStrategy(Strategy):
    def __init__(self, executor,risk_manager,predict_time=None):
        self.executor = executor
        self.risk_manager = risk_manager
        self.portfolio = self.executor.portofolio
        self.portfolio_contain_size = 20
        self.predict_time = predict_time
        pass

    def compute_allocation(self,data,pipeline_data):

        xq_profolio_real = self.portfolio[self.portfolio['short_time'].isnull()]
        shorts = {}
        for index, value in xq_profolio_real.iterrows():
            shorts[index] = 0.0

        df = pipeline_data.sort_values(axis=0, by='predict', ascending=False)

        print df[:self.portfolio_contain_size],
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
        '''
        universe = make_china_equity_universe(
            target_size=3000,
            mask=default_china_equity_universe_mask([RISK_BENCHMARK]),
            max_group_weight=0.01,
            smoothing_func=lambda f: f.downsample('month_start'),

        )
        private_universe = private_universe_mask(self.portfolio.index)  # 把当前组合的stock 包含在universe中
        '''
        last_price = USEquityPricing.close.latest >= 1.0  # 大于1元
        sector = get_sector()
        sector_filter = sector != 0.0
        universe = last_price & sector_filter
        # print "universe:",universe
        # Instantiate ranked factors
        returns = Returns(inputs=[USEquityPricing.close],mask=universe,window_length=2)
        risk_beta = 0.66 * RollingLinearRegressionOfReturns(
            target=symbol(RISK_BENCHMARK),
            returns_length=5,
            regression_length=21,
            # mask=long_short_screen
            mask=(universe),
        ).beta + 0.33 * 1.0
        returns.window_safe = True
        risk_beta.window_safe = True
        predict  = RNNPredict(universe,source='predict.csv',trigger_date=self.predict_time)  # 进行回顾
        columns = {
            'predict':predict,
            'market_beta': risk_beta,
            'sector': sector,
        }
        return columns,universe
