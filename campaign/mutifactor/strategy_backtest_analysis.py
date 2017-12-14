# -*- coding: utf-8 -*-

from zipline.pipeline import Pipeline, engine
from zipline.pipeline.factors import AverageDollarVolume, Returns
from zipline.pipeline.engine import (
    ExplodingPipelineEngine,
    SimplePipelineEngine,
)
from zipline.pipeline.factors import CustomFactor
from zipline.algorithm import TradingAlgorithm
from zipline.data.bundles.core import load
from zipline.data.data_portal import DataPortal
from zipline.finance.trading import TradingEnvironment
from zipline.finance.execution import MarketOrder
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.utils.calendars import get_calendar
from zipline.utils.factory import create_simulation_parameters
from zipline.utils.cli import Date, Timestamp
from me.pipeline.filters.universe import make_china_equity_universe, default_china_equity_universe_mask, \
    private_universe_mask

from zipline.api import (
    attach_pipeline,
    date_rules,
    time_rules,
    pipeline_output,
    record,
    schedule_function,
    symbol,
    get_datetime,
    order,
    order_target_percent
)
import pandas as pd
import os
import re

DEFAULT_CAPITAL_BASE = 1e5

from zipline.data.bundles import register
from zipline.data.bundles.viadb import viadb
import numpy as np
import pandas as pd

from zipline.pipeline.factors import CustomFactor, Returns, Latest, RSI
from me.pipeline.classifiers.tushare.sector import get_sector, RandomUniverse, get_sector_class, get_sector_by_onehot
from zipline.pipeline.factors import CustomFactor
from me.pipeline.factors.boost import Momentum
import tushare as ts

from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import Returns, AverageDollarVolume

from me.pipeline.factors.tsfactor import Fundamental
from me.pipeline.filters.universe import make_china_equity_universe, default_china_equity_universe_mask, \
    private_universe_mask
from zipline.utils.cli import Date, Timestamp

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(1335)  # for reproducibility

g_models = [LinearRegression(), Ridge(), Lasso(), ElasticNet(), KNeighborsRegressor(), DecisionTreeRegressor(), SVR(),
            RandomForestRegressor(), AdaBoostRegressor(), GradientBoostingRegressor()]
g_idx = 0


NUM_LONG_POSITIONS = 20
NUM_SHORT_POSITIONS = 20


c, _ = get_sector_class()
ONEHOTCLASS = tuple(c)



class ILLIQ(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = int(252)

    def compute(self, today, assets, out, close, volume):
        window_length = len(close)
        _rets = np.abs(pd.DataFrame(close, columns=assets).pct_change()[1:])
        _vols = pd.DataFrame(volume, columns=assets)[1:]
        out[:] = (_rets / _vols).mean().values




def BasicFactorRegress(inputs, window_length, mask, n_fwd_days, algo_mode=None, cross=True):
    class BasicFactorRegress(CustomFactor):
        # params = {'trigger_date': None, }
        init = False

        def __shift_mask_data(self, X, Y, n_fwd_days=1):
            # Shift X to match factors at t to returns at t+n_fwd_days (we want to predict future returns after all)
            shifted_X = np.roll(X, n_fwd_days, axis=0)
            # Slice off rolled elements
            X = shifted_X[n_fwd_days:]
            Y = Y[n_fwd_days:]
            n_time, n_stocks, n_factors = X.shape
            # Flatten X
            X = X.reshape((n_time * n_stocks, n_factors))
            Y = Y.reshape((n_time * n_stocks))
            return X, Y

        def __get_last_values(self, input_data):
            last_values = []
            for dataset in input_data:
                last_values.append(dataset[-1])
            return np.vstack(last_values).T

        def compute(self, today, assets, out, returns, *inputs):
            if (not self.init):
                self.clf = algo_mode
                X = np.dstack(inputs)  # (time, stocks, factors)  按时间组织了
                Y = returns  # (time, stocks)
                X, Y = self.__shift_mask_data(X, Y, n_fwd_days)  # n天的数值被展开成1维的了- 每个factor 按天展开
                X = np.nan_to_num(X)
                Y = np.nan_to_num(Y)
                if cross == True:
                    quadratic_featurizer = PolynomialFeatures(interaction_only=True)
                    X = quadratic_featurizer.fit_transform(X)

                self.clf.fit(X, Y)
                # self.init = True
            last_factor_values = self.__get_last_values(inputs)
            last_factor_values = np.nan_to_num(last_factor_values)

            out[:] = self.clf.predict(last_factor_values)

    return BasicFactorRegress(inputs=inputs, window_length=window_length, mask=mask)


def make_pipeline(asset_finder, algo_mode):
    hs300 = ts.get_hs300s()['code']
    private_universe = private_universe_mask(hs300.tolist(), asset_finder=asset_finder)
    # private_universe = private_universe_mask( ['000001','000002','000005'],asset_finder=asset_finder)
    ######################################################################################################
    returns = Returns(inputs=[USEquityPricing.close], window_length=5, mask=private_universe)  # 预测一周数据
    ######################################################################################################
    pe = Fundamental(mask=private_universe, asset_finder=asset_finder).pe
    pb = Fundamental(mask=private_universe, asset_finder=asset_finder).pb
    bvps = Fundamental(mask=private_universe, asset_finder=asset_finder).bvps
    market = Fundamental(mask=private_universe, asset_finder=asset_finder).outstanding
    totals = Fundamental(mask=private_universe, asset_finder=asset_finder).totals
    totalAssets = Fundamental(mask=private_universe, asset_finder=asset_finder).totalAssets
    fixedAssets = Fundamental(mask=private_universe, asset_finder=asset_finder).fixedAssets
    esp = Fundamental(mask=private_universe, asset_finder=asset_finder).esp
    rev = Fundamental(mask=private_universe, asset_finder=asset_finder).rev
    profit = Fundamental(mask=private_universe, asset_finder=asset_finder).profit
    gpr = Fundamental(mask=private_universe, asset_finder=asset_finder).gpr
    npr = Fundamental(mask=private_universe, asset_finder=asset_finder).npr

    rev10 = Returns(inputs=[USEquityPricing.close], window_length=10, mask=private_universe)
    vol10 = AverageDollarVolume(window_length=20, mask=private_universe)
    rev20 = Returns(inputs=[USEquityPricing.close], window_length=20, mask=private_universe)
    vol20 = AverageDollarVolume(window_length=20, mask=private_universe)
    rev30 = Returns(inputs=[USEquityPricing.close], window_length=30, mask=private_universe)
    vol30 = AverageDollarVolume(window_length=20, mask=private_universe)

    illiq22 = ILLIQ(window_length=22, mask=private_universe)
    illiq5 = ILLIQ(window_length=5, mask=private_universe)

    rsi5 = RSI(window_length=5, mask=private_universe)
    rsi22 = RSI(window_length=22, mask=private_universe)

    mom5 = Momentum(window_length=5, mask=private_universe)
    mom22 = Momentum(window_length=22, mask=private_universe)

    sector = get_sector(asset_finder=asset_finder, mask=private_universe)
    ONEHOTCLASS, sector_indict_keys = get_sector_by_onehot(asset_finder=asset_finder, mask=private_universe)

    pipe_columns = {

        'pe': pe.zscore(groupby=sector).downsample('month_start'),
        'pb': pb.zscore(groupby=sector).downsample('month_start'),
        'bvps': bvps.zscore(groupby=sector).downsample('month_start'),
        'market_cap': market.zscore(groupby=sector).downsample('month_start'),
        'totals': totals.zscore(groupby=sector).downsample('month_start'),
        'totalAssets': totalAssets.zscore(groupby=sector).downsample('month_start'),
        'fixedAssets': fixedAssets.zscore(groupby=sector).downsample('month_start'),
        'esp': esp.zscore(groupby=sector).downsample('month_start'),
        'rev': rev.zscore(groupby=sector).downsample('month_start'),
        'profit': profit.zscore(groupby=sector).downsample('month_start'),
        'gpr': gpr.zscore(groupby=sector).downsample('month_start'),
        'npr': npr.zscore(groupby=sector).downsample('month_start'),
        'vol10': vol10.zscore(groupby=sector).downsample('week_start'),
        'rev10': rev10.zscore(groupby=sector).downsample('week_start'),
        'vol20': vol20.zscore(groupby=sector).downsample('week_start'),
        'rev20': rev20.zscore(groupby=sector).downsample('week_start'),
        'vol30': vol30.zscore(groupby=sector).downsample('week_start'),
        'rev30': rev30.zscore(groupby=sector).downsample('week_start'),

        'ILLIQ5': illiq5.zscore(groupby=sector).downsample('week_start'),
        'ILLIQ22': illiq22.zscore(groupby=sector).downsample('week_start'),

        'mom5': mom5.zscore(groupby=sector).downsample('week_start'),
        'mom22': mom22.zscore(groupby=sector).downsample('week_start'),

        'rsi5': rsi5.zscore(groupby=sector).downsample('week_start'),
        'rsi22': rsi22.zscore(groupby=sector).downsample('week_start'),

    }

    from collections import OrderedDict
    factors_pipe = OrderedDict()

    factors_pipe['Returns'] = returns
    factors_pipe['Returns'].window_safe = True

    sort_keys = sorted(pipe_columns)
    for key in sort_keys:
        factors_pipe[key] = pipe_columns[key]
        factors_pipe[key].window_safe = True

    i = 0
    for c in ONEHOTCLASS:
        c.window_safe = True
        factors_pipe[sector_indict_keys[i]] = c
        # print (c,sector_indict_keys[i])
        i += 1

    predict = BasicFactorRegress(inputs=factors_pipe.values(), window_length=252, mask=private_universe,
                                 n_fwd_days=5,
                                 algo_mode=algo_mode,
                                 cross=False)
    predict_rank = predict.rank(mask=private_universe)

    longs = predict_rank.top(NUM_LONG_POSITIONS)
    shorts = predict_rank.bottom(NUM_SHORT_POSITIONS)
    long_short_screen = (longs | shorts)
    # TODO sector onehot
    pipe_final_columns = {
        'Predict Factor': predict.downsample('week_start'),
        'longs': longs.downsample('week_start'),
        'shorts': shorts.downsample('week_start'),
        'predict_rank': predict_rank.downsample('week_start'),
    }
    pipe = Pipeline(columns=pipe_final_columns,
                    screen=long_short_screen, )
    return pipe


############################################# bundle #############################################
equities1 = {}
register(
    'my-db-bundle',  # name this whatever you like
    viadb(equities1),
    calendar='SHSZ'
)
bundle = 'my-db-bundle'
bundle_timestamp = pd.Timestamp.utcnow()
environ = os.environ
bundle_data = load(
    bundle,
    environ,
    bundle_timestamp,
)
prefix, connstr = re.split(
    r'sqlite:///',
    str(bundle_data.asset_finder.engine.url),
    maxsplit=1,
)
# print prefix, connstr
if prefix:
    raise ValueError(
        "invalid url %r, must begin with 'sqlite:///'" %
        str(bundle_data.asset_finder.engine.url),
    )

############################################# trading_environment #############################################
trading_calendar = get_calendar("SHSZ")
trading_environment = TradingEnvironment(bm_symbol=None,
                                         exchange_tz="Asia/Shanghai",
                                         trading_calendar=trading_calendar,
                                         asset_db_path=connstr)

############################################# choose_loader #############################################

pipeline_loader = USEquityPricingLoader(
    bundle_data.equity_daily_bar_reader,
    bundle_data.adjustment_reader,
)


def choose_loader(column):
    if column in USEquityPricing.columns:
        return pipeline_loader
    raise ValueError(
        "No PipelineLoader registered for column %s." % column
    )


###################################### data ###################################

first_trading_day = \
    bundle_data.equity_minute_bar_reader.first_trading_day
data = DataPortal(
    trading_environment.asset_finder, trading_calendar,
    first_trading_day=first_trading_day,
    equity_minute_reader=bundle_data.equity_minute_bar_reader,
    equity_daily_reader=bundle_data.equity_daily_bar_reader,
    adjustment_reader=bundle_data.adjustment_reader,
)
################################## sim_params
capital_base = DEFAULT_CAPITAL_BASE
start = '2017-1-1'
end   = '2017-11-30'
sim_params = create_simulation_parameters(
    capital_base=capital_base,
    start=Date(tz='utc', as_timestamp=True).parser(start),
    end=Date(tz='utc', as_timestamp=True).parser(end),
    data_frequency='daily',
    trading_calendar=trading_calendar,
)


#######################################################################
def rebalance(context, data):
    # print ("rebalance:",get_datetime())
    # print context.pipeline_data
    pipeline_data = context.pipeline_data
    keys = list(context.posset)
    for asset in keys:
        if data.can_trade(asset):
            # print("flattern:",asset)
            order_target_percent(asset=asset, target=0.0, style=MarketOrder())
            del context.posset[asset]

    for asset, value in pipeline_data[pipeline_data['shorts'] == True].iterrows():
        if data.can_trade(asset):
            # print("short:", asset,- 1.0/NUM_SHORT_POSITIONS)
            order_target_percent(asset=asset, target=-1.0 / (2 * NUM_SHORT_POSITIONS), style=MarketOrder())
            context.posset[asset] = False

    for asset, value in pipeline_data[pipeline_data['longs'] == True].iterrows():
        # print type(label),value
        if data.can_trade(asset):
            # print("long:", asset,1.0/NUM_LONG_POSITIONS)
            order_target_percent(asset=asset, target=1.0 / (2 * NUM_LONG_POSITIONS), style=MarketOrder())
            context.posset[asset] = True
    # print("rebalance over")
    # if context.rbcnt == 0:
    #	pipeline_data.to_csv(str(context.rbcnt) + "-rb.csv", encoding="utf-8")
    context.rbcnt += 1


def initialize(context):
    model = g_models[g_idx]
    print("hello world --- :", g_idx, model)
    attach_pipeline(make_pipeline(asset_finder=None, algo_mode=model), 'my_pipeline')
    schedule_function(rebalance, date_rules.week_start(days_offset=0), half_days=True)
    context.posset = {}
    context.rbcnt = 0
    # record my portfolio variables at the end of day


def handle_data(context, data):
    # print symbol('000001'),data.current(symbol('000001'), 'price')
    pass


def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')
    # print("pipeline_data",type(context.pipeline_data))
    # print(context.portfolio)
    pass


#################################################################################################################################################
for i in range(0, len(g_models)):
    # for i in range(0,4):
    g_idx = i
    algor_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data,
                                 before_trading_start=before_trading_start,
                                 sim_params=sim_params,
                                 env=trading_environment,
                                 data_frequency='daily',
                                 get_pipeline_loader=choose_loader,
                                 )

    result = algor_obj.run(data)
    # result.to_csv("result.csv", encoding="utf-8")
    # print(result)

    # for index,values in  result.iterrows():
    #     print index 
    #     for t in values['transactions']:
    #         print t
    #     print "++++++++++++++++++++++++++++++++++++++++++++"

    # for x in result.columns:
    #     print x
    # print (result.tail(10))
    print (result.pnl.cumsum()[-1], result.algorithm_period_return[-1])
    # import matplotlib
    # matplotlib.use('agg')
    # import pyfolio as pf
    # returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(result)
    # print(returns)
    # print(returns.cumsum())
