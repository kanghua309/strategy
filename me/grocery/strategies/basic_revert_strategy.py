# -*- coding: utf-8 -*-

from zipline.pipeline.data import USEquityPricing
from zipline.api import (
    symbol,
    get_datetime,
)
from datetime import timedelta, datetime
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import RollingLinearRegressionOfReturns,Latest
from me.pipeline.classifiers.tushare.sector import get_sector
from me.pipeline.factors.boost import HurstExp,Beta
from me.pipeline.filters.universe import make_china_equity_universe, default_china_equity_universe_mask, \
    private_universe_mask

from strategy import Strategy


risk_benchmark = '000001'
class RevertStrategy(Strategy):
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
        # print pipeline_data.loc['000018']
        # context.xueqiuLive.login()
        print "Rebalance - Current xq profolio"
        print len(self.portfolio), self.portfolio

        xq_profolio_real = self.portfolio[self.portfolio['short_time'].isnull()]
        remove_dict = self.__check_stop_limit(data)
        print "remove_stock for stop:", remove_dict
        _remove     = self.__check_expired_limit(data)
        remove_dict.update(_remove)  # TODO
        print "remove_stock for expire:", remove_dict
        profolio_hold_index = xq_profolio_real.index.difference(remove_dict)
        print "-----------------------------------sell first------------------------------------------"
        for index, row in pipeline_data.ix[profolio_hold_index].iterrows():  # 应该有很hold里的在data中找不到，没关系，忽略之
            hurst = row.hurst
            vbeta = row.volume_pct_beta
            pbeta = row.price_pct_beta
            if hurst <= 0.2:
                if vbeta > 0 and vbeta < pbeta:
                    print("++++++++++++++++++++++++++Info sell sym(%s) for mean revert at all" % index)
                    remove_dict[index] = 0.0

        profolio_hold_index = profolio_hold_index.difference(remove_dict)
        pools = pipeline_data.index.difference(xq_profolio_real.index)
        print "profolio_hold_index before buy:", profolio_hold_index
        print "-----------------------------------buy last------------------------------------------"
        for index, row in pipeline_data.ix[pools].iterrows():
            hurst = row.hurst
            vbeta = row.volume_pct_beta
            pbeta = row.price_pct_beta
            if hurst <= 0.2:
                if vbeta < 0 and vbeta < pbeta:  # 先买均值回归的！ 安全！！！
                    print("==========================Info buy sym(%s) for mean revert" % (index))
                    profolio_hold_index = profolio_hold_index.insert(0, index)
            if len(profolio_hold_index) == self.portfolio_contain_size:
                break
                # print "profolio_hold_index:",profolio_hold_index
        print "profolio_hold_index after buy:", profolio_hold_index, len(profolio_hold_index)
        profolio_hold = pipeline_data.loc[profolio_hold_index]
        weights = self.risk_manager.optimalize(profolio_hold,{'ALPHA':'volume_pct_beta','BETA':'market_beta','SECTOR':'sector'})
        return remove_dict,weights


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
            target_size=2000,
            mask=default_china_equity_universe_mask([risk_benchmark]),
            max_group_weight=0.01,
            smoothing_func=lambda f: f.downsample('month_start'),

        )
        last_price = USEquityPricing.close.latest >= 1.0  #大于1元
        #last_price = Latest().latest >= 1.0

        private_universe = private_universe_mask(self.portfolio.index)
        universe = universe & last_price | private_universe
        hurst = HurstExp(window_length=int(252 * 0.25), mask=universe)
        sector = get_sector()

        #top = hurst.top(2, groupby=sector)
        bottom = hurst.bottom(2, groupby=sector)
        # universe = (top | bottom) | private_universe
        universe = (bottom) & (sector != 0) | private_universe
        combined_rank = (
            hurst.rank(mask=universe)
        )
        pct_beta = Beta(window_length=21, mask=(universe))
        risk_beta = 0.66 * RollingLinearRegressionOfReturns(
            target=symbol(risk_benchmark),  # sid(8554),
            returns_length=6,
            regression_length=21,
            # mask=long_short_screen
            mask=(universe),
        ).beta + 0.33 * 1.0

        columns= {
                    'hurst': hurst.downsample('week_start'),
                    'price_pct_beta' : pct_beta.pbeta,
                    'volume_pct_beta': pct_beta.vbeta,
                    'sector': sector.downsample('week_start'),
                    'market_beta': risk_beta,
                    'rank':combined_rank,
                    #'testrank':hurst.rank(mask=universe)
                 }
        return columns,universe
