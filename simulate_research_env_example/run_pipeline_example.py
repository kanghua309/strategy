from zipline.pipeline import Pipeline, engine
from zipline.pipeline.factors import AverageDollarVolume, Returns
from zipline.pipeline.engine import (
    SimplePipelineEngine,
)
from zipline.data.bundles.core import load
from zipline.finance.trading import TradingEnvironment
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.utils.calendars import get_calendar
from zipline.utils.cli import Date, Timestamp

import pandas as pd
import os
import re
DEFAULT_CAPITAL_BASE = 1e5

from zipline.pipeline.factors import CustomFactor
from zipline.data.bundles import register
from zipline.data.bundles.viadb import viadb


# custom factor
class DV2(CustomFactor):
    inputs = [USEquityPricing.close,USEquityPricing.high, USEquityPricing.low]
    window_length = 3
    def compute(self, today, assets, out, close, high, low):
        dv = 100 * ((close / (0.5 * ( high + low)))-1)
        dv2 = (dv[1:] + dv[:-1]) / 2
        out[:] = dv2[-1]

N = 10
def make_pipeline():
    dollar_volume = AverageDollarVolume(window_length=1)
    high_dollar_volume = dollar_volume.percentile_between(N, 100)
    recent_returns = Returns(window_length=N, mask=high_dollar_volume)
    low_returns = recent_returns.percentile_between(0, 10)
    high_returns = recent_returns.percentile_between(N, 100)
    dv2 = DV2()
    pipe_columns = {
        'low_returns': low_returns,
        'high_returns': high_returns,
        'recent_returns': recent_returns,
        'dollar_volume': dollar_volume,
        'dv2':dv2
    }
    pipe_screen = (low_returns | high_returns)
    pipe = Pipeline(columns=pipe_columns, screen=pipe_screen)
    return pipe

my_pipe = make_pipeline()

############################################# china astock bundle ##############################################
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

#data_frequency = 'daily',
#capital_base = DEFAULT_CAPITAL_BASE

start = '2015-9-1'  # 必须在国内交易日
end   = '2017-9-9'  # 必须在国内交易日

#print Date(tz='utc', as_timestamp=True).parser(start)
#perf_tracker = None
# Pull in the environment's new AssetFinder for quick reference
#print trading_calendar.all_sessions

your_engine = SimplePipelineEngine(get_loader=choose_loader, calendar=trading_calendar.all_sessions, asset_finder=trading_environment.asset_finder)
result = your_engine.run_pipeline(my_pipe,Date(tz='utc', as_timestamp=True).parser(start),Date(tz='utc', as_timestamp=True).parser(end))
print result

