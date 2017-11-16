# -*- coding: utf-8 -*-

import logging

import click
import gym
import numpy as np
from keras.models import load_model

log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)

import gym_trading  #必须引入才自动注册

@click.command()
@click.option(
    '-s',
    '--symbol',
    default='000001',
    show_default=True,
    help='given stock code ',
)
@click.option(
    '-b',
    '--begin',
    default='2017-01-01',
    show_default=True,
    help='The begin date of the train.',
)
@click.option(
    '-e',
    '--end',
    default='2017-10-31',
    show_default=True,
    help='The end date of the train.',
)
@click.option(
    '-d',
    '--days',
    type=int,
    default=30,
    help='train days',
)
@click.option(
    '--plot/--no-plot',
    # default=os.name != "nt",
    is_flag=True,
    default=True,
    help="render when training"
)
@click.option(
    '-m',
    '--model_path',
    default='000001.model',
    show_default=True,
    help='trained model save path.',
)

@click.option(
    '-r',
    '--random',
    is_flag=True,
    default=False,
    help="random range"
)


def execute(symbol, begin, end, days, plot, model_path,random):
    print model_path
    model = load_model(model_path)
    env = gym.make('trading-v0').env
    env.initialise(symbol=symbol, start=begin, end=end, days=days, random = random)
    state_size = env.observation_space.shape[0]
    state = env.reset()
    done = False
    while not done:
        state = state.reshape(1, state_size)
        # state = state.reshape(1, 1, state_size)
        qval = model.predict(state, batch_size=1)
        action = (np.argmax(qval))
        state, _, done, info = env.step(action)

        # log.info("%s,%s,%s,%s",state, _, done, info)
        # log.info("\n%s", env.sim.to_df())
        if plot:
           env.render()


if __name__ == "__main__":
    execute()
