# -*- coding: utf-8 -*-

from zipline.pipeline.data import USEquityPricing
from zipline.api import (
    symbol,
    get_datetime,
)
from datetime import timedelta, datetime
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import RollingLinearRegressionOfReturns,Latest,Returns
from me.pipeline.classifiers.tushare.sector import get_sector
from me.pipeline.factors.boost import HurstExp,Slope
from me.pipeline.filters.universe import make_china_equity_universe, default_china_equity_universe_mask, \
    private_universe_mask
from me.pipeline.factors.risk import Markowitz

from .strategy import Strategy


risk_benchmark = '000001'

STOP_LOSS = 0.1  #止损
STOP_WIN  = 0.1  #止盈
class RevertStrategy(Strategy):
    def __init__(self, executor,risk_manager):
        self.executor = executor
        self.risk_manager = risk_manager
        self.portfolio = self.executor.portofolio
        self.portfolio_contain_size = 19 #股票组合容量20，除去一个placeholder ，共19个
        pass

    def __check_stop_limit(self,data):
        profolio = self.portfolio[self.portfolio['short_time'].isnull()].keep_price
        stop_dict = {}
        for index, value in profolio.iteritems():
            keep_price = profolio[index]
            current_price = data.current(symbol(index), 'price')
            # print "Rebalance - index, keep_price, current_price"
            if keep_price / current_price > 1 + STOP_WIN:
                print ("%s has down to stop limit, sell it - for %s,%s " % (index, keep_price, current_price))
                stop_dict[index] = 0.0
            if keep_price / current_price < 1 - STOP_LOSS:
                print ("%s has up to expected price , sell it - for %s,%s" % (index, keep_price, current_price))
                stop_dict[index] = 0.0
        return stop_dict

    def __check_expired_limit(self,data):
        profolio = self.portfolio[self.portfolio['short_time'].isnull()].long_time
        stop_dict = {}
        for index, value in profolio.iteritems():
            lastdt = profolio[index]
            # print "Rebalance - index, keep_price, current_price"
            if datetime.now() - lastdt > timedelta(weeks=1):
                print ("%s has expired , sell it - for %s,%s" % (index, datetime.now(), lastdt))
                stop_dict[index] = 0.0
        return stop_dict

    def compute_allocation(self,data,pipeline_data):
        # print pipeline_data.loc['000018']
        # context.xueqiuLive.login()
        print ("Rebalance - Current xq profolio")
        #print len(self.portfolio), self.portfolio

        xq_profolio_real = self.portfolio[self.portfolio['short_time'].isnull()]
        remove_dict = self.__check_stop_limit(data)
        print ("Rebalance - remove_stock for stop:", remove_dict)
        _remove     = self.__check_expired_limit(data)
        remove_dict.update(_remove)  # TODO
        print ("Rebalance - remove_stock for expire:", remove_dict)
        profolio_hold_index = xq_profolio_real.index.difference(remove_dict)
        for index, row in pipeline_data.ix[profolio_hold_index].iterrows():  # 应该有很hold里的在data中找不到，没关系，忽略之
            hurst = row.hurst
            vslope = row.volume_pct_slope
            pslope = row.price_pct_slope
            if hurst <= 0.3:  #均值反转特性
                if vslope > 0 and vslope < pslope:
                    print("++++++++++++++++++++++++++Info sell sym(%s) for mean revert at all" % index)
                    remove_dict[index] = 0.0

        profolio_hold_index = profolio_hold_index.difference(remove_dict)
        print (len(profolio_hold_index))
        pools = pipeline_data.index.difference(xq_profolio_real.index)
        # print "profolio_hold_index before buy:", profolio_hold_index
        for index, row in pipeline_data.ix[pools].iterrows():
            if len(profolio_hold_index) >= self.portfolio_contain_size:
                break
            hurst = row.hurst
            vslope = row.volume_pct_slope
            pslope = row.price_pct_slope
            if hurst <= 0.3:  #均值反转特性
                if vslope < 0 and vslope < pslope:  # 有加速效应
                    print("++++++++++++++++++++++++++Info buy sym(%s) for mean revert" % (index))
                    profolio_hold_index = profolio_hold_index.insert(0, index)

                # print "profolio_hold_index:",profolio_hold_index
        print ("Rebalance - Profolio_hold_index now:",profolio_hold_index)
        profolio_hold = pipeline_data.loc[profolio_hold_index]
        weights = self.risk_manager.optimalize(profolio_hold,{'ALPHA':'volume_pct_slope','BETA':'market_beta','SECTOR':'sector',"RETURNS":'returns'})  #作为参数优化的必备项
        return remove_dict,weights


    def trade(self,shorts,longs):
        print ("do sell .....")
        #self.executor.orders(shorts)
        print ("do buy .....")
        #self.executor.orders(longs)
        pass

    def portfolio(self):
        raise NotImplementedError()

    def pipeline_columns_and_mask(self):
        universe = make_china_equity_universe(
            target_size=2000,
            mask=default_china_equity_universe_mask([risk_benchmark]),
            max_group_weight=0.01,
            smoothing_func=lambda f: f.downsample('month_start'),
        )
        private_universe = private_universe_mask(self.portfolio.index) #把当前组合的stock 包含在universe中

        last_price = USEquityPricing.close.latest >= 1.0  #大于1元
        universe = universe & last_price | private_universe
        hurst = HurstExp(window_length=int(252 * 0.25), mask=universe) #判断动量或反转特性指标
        sector = get_sector()
        #combined_rank = (
        #    hurst.rank(mask=universe)
        #)
        pct_slope = Slope(window_length=21, mask=(universe))  #量和价格加速度
        risk_beta = 0.66 * RollingLinearRegressionOfReturns(
            target=symbol(risk_benchmark),
            returns_length=5,
            regression_length=21,
            mask=(universe),
        ).beta + 0.33 * 1.0
        returns = Returns(inputs=[USEquityPricing.close],
                mask=universe,window_length=2)
        #returns.window_safe = True
        #risk_beta.window_safe = True
        #m = Markowitz(inputs=[returns,risk_beta],window_length=6,mask=universe)
        columns= {
                    'hurst': hurst.downsample('week_start'),
                    'price_pct_slope' : pct_slope.pslope,
                    'volume_pct_slope': pct_slope.vslope,
                    'sector': sector.downsample('month_start'),
                    'market_beta': risk_beta,
                    'returns':returns,
                 }
        return columns,universe
