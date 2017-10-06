from zipline.pipeline import Pipeline, engine
from zipline.pipeline.factors import AverageDollarVolume, Returns
from zipline.pipeline.engine import (
    ExplodingPipelineEngine,
    SimplePipelineEngine,
)
from zipline.algorithm import TradingAlgorithm
from zipline.data.bundles.core import load
from zipline.data.data_portal import DataPortal
from zipline.finance.trading import TradingEnvironment
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.utils.calendars import get_calendar
from zipline.utils.factory import create_simulation_parameters
from zipline.utils.cli import Date, Timestamp
from zipline.api import (
    attach_pipeline,
    date_rules,
    time_rules,
    pipeline_output,
    record,
    schedule_function,
    symbol,
    get_datetime,
    order
)
import pandas as pd
import os
import re
DEFAULT_CAPITAL_BASE = 1e5

from zipline.data.bundles import register
from zipline.data.bundles.viadb import viadb


N = 10
def make_pipeline():
    dollar_volume = AverageDollarVolume(window_length=1)
    high_dollar_volume = dollar_volume.percentile_between(N, 100)
    recent_returns = Returns(window_length=N, mask=high_dollar_volume)
    low_returns = recent_returns.percentile_between(0, 10)
    high_returns = recent_returns.percentile_between(N, 100)
    pipe_columns = {
        'low_returns': low_returns,
        'high_returns': high_returns,
        'recent_returns': recent_returns,
        'dollar_volume': dollar_volume
    }
    pipe_screen = (low_returns | high_returns)
    pipe = Pipeline(columns=pipe_columns, screen=pipe_screen)
    return pipe

############################################# bundle #############################################
equities1={}
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
print prefix, connstr
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
start = '2015-9-1'
end   = '2016-9-1'
sim_params = create_simulation_parameters(
             capital_base= capital_base,
             start=Date(tz='utc', as_timestamp=True).parser(start),
             end=Date(tz='utc', as_timestamp=True).parser(end),
             data_frequency='daily',
             trading_calendar=trading_calendar,
           )

#######################################################################
def rebalance(context, data):
    print("rebalance")


def initialize(context):
    print("hello world")
    attach_pipeline(make_pipeline(), 'my_pipeline')
    schedule_function(rebalance, date_rules.week_end(days_offset=0), half_days=True)
    # record my portfolio variables at the end of day

def handle_data(context, data):
    print symbol('000001'),data.current(symbol('000001'), 'price')
    order(symbol('000001'), 1)

def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')
    print("pipeline_data",context.pipeline_data)
    pass

algor_obj = TradingAlgorithm(initialize=initialize, handle_data=handle_data,before_trading_start = before_trading_start,
                             sim_params=sim_params,
                             env=trading_environment,
                             data_frequency = 'daily',
                             get_pipeline_loader = choose_loader,
                             )
perf_manual = algor_obj.run(data)

print perf_manual