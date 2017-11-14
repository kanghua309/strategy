"""
A simple Pipeline algorithm that longs the top 3 stocks by RSI and shorts
the bottom 3 each day.
"""
from six import viewkeys
from zipline.api import (
    attach_pipeline,
    date_rules,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    symbol,
)
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import RSI



def make_pipeline(context):
    print context.optim_history_depth
    rsi = RSI(window_length=context.optim_history_depth)
    return Pipeline(
        columns={
            'longs': rsi.top(context.optim_top_num),
            'shorts': rsi.bottom(context.optim_bottom_num),
        },
    )


def rebalance(context, data):

    #############################################################################
    # Pipeline data will be a dataframe with boolean columns named 'longs' and
    # 'shorts'.
    pipeline_data = context.pipeline_data
    all_assets = pipeline_data.index

    longs = all_assets[pipeline_data.longs]
    shorts = all_assets[pipeline_data.shorts]

    record(universe_size=len(all_assets))

    # Build a 2x-leveraged, equal-weight, long-short portfolio.
    one_third = 1.0 / context.optim_leveraged
    for asset in longs:
        order_target_percent(asset, one_third)

    for asset in shorts:
        order_target_percent(asset, -one_third)

    # Remove any assets that should no longer be in our portfolio.
    portfolio_assets = longs | shorts
    positions = context.portfolio.positions
    for asset in viewkeys(positions) - set(portfolio_assets):
        # This will fail if the asset was removed from our portfolio because it
        # was delisted.
        if data.can_trade(asset):
            order_target_percent(asset, 0)


def initialize(context):
    context.optim_leveraged = 3.0
    context.optim_history_depth = 15
    context.optim_top_num = 3
    context.optim_bottom_num = 3

    UseParams = False
    try:
        Nparams = len(context.algo_params)
        if Nparams == 4:
            UseParams = True
        else:
            print 'len context.algo_params is', Nparams, ' expecting 4'
    except Exception as e:
        print 'context.params not passed', e
    if UseParams:
        context.optim_leverage = context.algo_params['leverage'][0]
        context.optim_history_depth = context.algo_params['history_depth'][0]
        context.optim_top_num = context.algo_params['top_num'][0]
        context.optim_bottom_num = context.algo_params['bottom_num'][0]
        print 'Setting Algo parameters via passed algo_params', context.optim_leverage, context.optim_history_depth, context.optim_top_num, context.optim_bottom_num

    attach_pipeline(make_pipeline(context), 'my_pipeline')
    schedule_function(rebalance, date_rules.every_day())


def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')