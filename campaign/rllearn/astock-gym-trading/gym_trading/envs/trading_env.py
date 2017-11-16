# -*- coding: utf-8 -*-

import logging
import tempfile

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from talib.abstract import *

from me.helper.research_env import Research

log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)

ret = lambda x, y: np.log(y / x)  # Log return
zscore = lambda x: (x - x.mean()) / x.std()  # zscore


class ZiplineEnvSrc(object):
    # Quandl-based implementation of a TradingEnv's data source.
    # Pulls data from Quandl, preps for use by TradingEnv and then
    # acts as data provider for each new episode.
    def __init__(self, symbol, start, end, days=252, scale=True):
        self.symbol = symbol
        self.days = days + 1
        self.start = start
        self.end = end

        log.info('getting data for %s from zipline bundle...', symbol)
        research = Research()

        panel = research.get_pricing([self.symbol], start, end, '1d', ['open', 'high', 'low', 'close', 'volume'])
        _df = panel.transpose(2, 1, 0).iloc[0]

        _df.volume.replace(0, 1, inplace=True)  # days shouldn't have zero volume..
        _df.dropna(axis=0, inplace=True)
        assert not np.any(np.isnan(_df))

        df = pd.DataFrame()
        ##########################
        close = _df['close'].values
        sma5 = SMA(_df, timeperiod=5)
        sma15 = SMA(_df, timeperiod=15)
        rsi = RSI(_df, timeperiod=5)
        atr = ATR(_df, timeperiod=5)

        df['Return'] = (_df.close - _df.close.shift()) / _df.close.shift()  # today return
        df['SMA5'] = zscore(sma5)
        df['SMA15'] = zscore(sma15)
        df['C-SMA5'] = zscore(close - sma5)
        df['SMA5-SMA15'] = zscore(sma5 - sma15)
        df['RSI'] = zscore(rsi)
        df['ATR'] = zscore(atr)
        df['VOL'] = zscore(_df.volume)
        df['CLOSE'] = zscore(_df.close)

        df.dropna(axis=0, inplace=True)

        self.min_values = df.min(axis=0)
        self.max_values = df.max(axis=0)
        self.data = df
        self.step = 0
        self.orgin_idx = 0
        self.prices = _df.close

    def reset(self,random):
        if random == True:
            self.idx = np.random.randint(low=0, high=len(self.data.index) - self.days)
        else:
            self.idx = len(self.data.index) - self.days
        self.step = 0
        self.orgin_idx = self.idx  # for render , so record it
        self.reset_start_day = str(self.data.index[self.orgin_idx -1 ])[:10]
        self.reset_end_day = str(self.data.index[self.orgin_idx + self.days -1 ])[:10]
        #print(self.reset_start_day,self.reset_end_day)


    def _step(self):
        obs = self.data.iloc[self.idx].as_matrix()
        self.idx += 1
        self.step += 1
        done = self.step >= self.days
        return obs, done


class TradingSim(object):
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps=1e-3, time_cost_bps=1e-4):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps
        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.mkt_nav = np.ones(self.steps)
        self.strat_retrns = np.ones(self.steps)
        self.posns = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.mkt_retrns = np.zeros(self.steps)

    def reset(self, train=True):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.mkt_nav.fill(1)
        self.strat_retrns.fill(0)
        self.posns.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.mkt_retrns.fill(0)

    def _step(self, action, retrn):
        bod_posn = 0.0 if self.step == 0 else self.posns[self.step - 1]
        bod_nav = 1.0 if self.step == 0 else self.navs[self.step - 1]
        mkt_nav = 1.0 if self.step == 0 else self.mkt_nav[self.step - 1]

        self.mkt_retrns[self.step] = retrn
        self.actions[self.step] = action

        self.posns[self.step] = action - 1
        self.trades[self.step] = self.posns[self.step] - bod_posn

        trade_costs_pct = abs(self.trades[self.step]) * self.trading_cost_bps
        self.costs[self.step] = trade_costs_pct + self.time_cost_bps
        reward = ((bod_posn * retrn) - self.costs[self.step])
        self.strat_retrns[self.step] = reward
        logging.debug(
            "debug ----- :retrn:%f,action:%d,bod_posn,%d,posn:%d,trades:%d,trade_costs_pct:%f,costs:%f,reward:%f" % (
                retrn, action,
                bod_posn,
                self.posns[self.step],
                self.trades[self.step],
                trade_costs_pct,
                self.costs[self.step],
                reward))

        if self.step != 0:
            self.navs[self.step] = bod_nav * (1 + self.strat_retrns[self.step - 1])
            self.mkt_nav[self.step] = mkt_nav * (1 + self.mkt_retrns[self.step - 1])

        info = {'reward': reward, 'nav': self.navs[self.step], 'costs': self.costs[self.step],
                'pos': self.posns[self.step]}
        self.step += 1
        return reward, info

    def to_df(self):
        """returns internal state in new dataframe """
        cols = ['action', 'bod_nav', 'mkt_nav', 'mkt_return', 'sim_return',
                'position', 'costs', 'trade']

        df = pd.DataFrame({'action': self.actions,  # today's action (from agent)
                           'bod_nav': self.navs,  # BOD Net Asset Value (NAV)
                           'mkt_nav': self.mkt_nav,  #
                           'mkt_return': self.mkt_retrns,
                           'sim_return': self.strat_retrns,
                           'position': self.posns,  # EOD position
                           'costs': self.costs,  # eod costs
                           'trade': self.trades},  # eod trade
                          columns=cols)
        return df


class TradingEnv(gym.Env):
    """This gym implements a simple trading environment for reinforcement learning.

    The gym provides daily observations based on real market data pulled
    from Quandl on, by default, the SPY etf. An episode is defined as 252
    contiguous days sampled from the overall dataset. Each day is one
    'step' within the gym and for each step, the algo has a choice:

    SHORT (0)
    FLAT (1)
    LONG (2)

    If you trade, you will be charged, by default, 10 BPS of the size of
    your trade. Thus, going from short to long costs twice as much as
    going from short to/from flat. Not trading also has a default cost of
    1 BPS per step. Nobody said it would be easy!

    At the beginning of your episode, you are allocated 1 unit of
    cash. This is your starting Net Asset Value (NAV). If your NAV drops
    to 0, your episode is over and you lose. If your NAV hits 2.0, then
    you win.

    The trading envs will track a buy-and-hold strategy which will act as
    the benchmark for the game.

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.inited = False;
        pass

    def initialise(self, symbol, start, end, days ,random = True):
        self.days = days
        self.src = ZiplineEnvSrc(symbol=symbol, start=start, end=end, days=self.days)
        self.sim = TradingSim(steps=self.days, trading_cost_bps=1e-4, time_cost_bps=1e-4)  # TODO FIX

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.src.min_values,
                                            self.src.max_values)
        self.reset()
        self.inited = True
        self.render_on = 0
        self.reset_count = 0
        self.random = random

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if self.inited == False: return
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        observation, done = self.src._step()
        yret = observation[0]  # RETURN
        reward, info = self.sim._step(action, yret)
        return observation, reward, done, info

    def _reset(self):
        if self.inited == False: return
        self.reset_count += 1
        self.src.reset(self.random)
        self.sim.reset()
        return self.src._step()[0]

    def _plot_trades(self):
        ####################################################################
        plt.subplot(2, 1, 1)
        p = self.src.prices[self.src.orgin_idx:]  # TODO
        p = p.reset_index(drop=True).head(self.days)
        p.plot(style='kx-', label='price')
        l = ['price']

        idx = (pd.Series(self.sim.trades) > 0)
        if idx.any():
            p[idx].plot(style='go')
            l.append('long')
        # colored line for short positions
        idx = (pd.Series(self.sim.trades) < 0)
        if idx.any():
            p[idx].plot(style='ro')
            l.append('short')

        plt.xlim([p.index[0], p.index[-1]])  # show full axis
        plt.legend(l, loc='upper right')
        plt.title('trades')
        plt.draw()
        ####################################################################
        l = []
        plt.subplot(2, 1, 2)
        plt.title('net value')
        pd.Series(self.sim.mkt_nav).plot(style='g')
        l.append("cumulative return by buy and hold ")
        pd.Series(self.sim.navs).plot(style='r')
        l.append("cumulative return by rl learning")
        plt.legend(l, loc='upper left')
        plt.draw()

        # plt.subplot(3, 1, 3)
        # pd.Series(self.sim.navs).plot(style='r')
        # plt.title('simulate net value')
        # plt.draw()

        return plt

    def _render(self, mode='human', close=False):
        if self.inited == False: return
        if self.render_on == 0:
            # self.fig = plt.figure(figsize=(10, 4))
            self.fig = plt.figure(figsize=(12, 6))
            self.render_on = 1
            plt.ion()

        plt.clf()
        self._plot_trades()
        plt.suptitle("Code: " + self.src.symbol + ' ' + \
                     "Round:" + str(self.reset_count) + "-" + \
                     "Step:" + str(self.src.idx - self.src.orgin_idx) + "  (" + \
                     "from:" + self.src.reset_start_day + " " + \
                     "to:" + self.src.reset_end_day + ")")
        plt.pause(0.001)
        return self.fig

    def run_strat(self, strategy, return_df=True):
        if self.inited == False: return
        """run provided strategy, returns dataframe with all steps"""
        observation = self.reset()
        done = False
        count = 0
        while not done:
            action = strategy(observation, self)  # call strategy
            observation, reward, done, info = self.step(action)
            count += 1
            print observation, reward, done, info, count

        return self.sim.to_df() if return_df else None

    def run_strats(self, strategy, episodes=1, write_log=True, return_df=True):
        if self.inited == False: return

        """ run provided strategy the specified # of times, possibly
            writing a log and possibly returning a dataframe summarizing activity.
    
            Note that writing the log is expensive and returning the df is moreso.  
            For training purposes, you might not want to set both.
        """
        logfile = None
        if write_log:
            logfile = tempfile.NamedTemporaryFile(delete=False)
            log.info('writing log to %s', logfile.name)
            need_df = write_log or return_df

        alldf = None

        for i in range(episodes):
            df = self.run_strat(strategy, return_df=need_df)
            if write_log:
                df.to_csv(logfile, mode='a')
                if return_df:
                    alldf = df if alldf is None else pd.concat([alldf, df], axis=0)

        return alldf