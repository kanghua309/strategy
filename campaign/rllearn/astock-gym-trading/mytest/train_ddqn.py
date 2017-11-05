# -*- coding: utf-8 -*-

import logging
import random
from collections import deque

import click
import gym
import gym_trading  #必须引入才自动注册

import numpy as np
import pandas as pd
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import Adam

PLOT_AFTER_ROUND = 1
log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)
class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.11
        self.epsilon_decay = 0.995
        self.learning_rate = 0.000003
        self.tau = .125
        self.batch_size = 64
        self.state_size = self.env.observation_space.shape[0]

        self.model = self.create_model()
        self.target_model = self.create_model()


    def create_model(self):
        model = Sequential()
        print 'state_shape:', self.env.observation_space.shape
        model.add(Dense(24, input_dim=self.state_size, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def act(self, state):
        """Acting Policy of the DQNAgent
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # return np.argmax(self.model.predict(state,batch_size=1))
        qval = self.model.predict(state)[0]
        #print "------------------------ qval:", np.argmax(qval),qval,state
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """Memory Management and training of the agent
        """
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done = self._get_batches()
        reward += (self.gamma
                   * np.logical_not(done)
                   * np.amax(self.model.predict(next_state), axis=1))
        q_target = self.target_model.predict(state)

        _ = pd.Series(action)
        one_hot = pd.get_dummies(_).as_matrix()
        action_batch = np.where(one_hot == 1)
        q_target[action_batch] = reward
        return self.model.fit(state, q_target,
                              batch_size=self.batch_size,
                              epochs=1,
                              verbose=False)

    def _get_batches(self):
        """Selecting a batch of memory
           Split it into categorical subbatches
           Process action_batch into a position vector
        """
        batch = np.array(random.sample(self.memory, self.batch_size))
        state_batch = np.concatenate(batch[:, 0]) \
            .reshape(self.batch_size, self.state_size)

        action_batch = batch[:, 1]

        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3]) \
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


    def target_train(self):
        weights = self.model.get_weights()
        #print "target train weight:",weights
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


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
    default='2017-11-01',
    show_default=True,
    help='The end date of the train.',
)

@click.option(
    '-d',
    '--days',
    type=int,
    default=100,
    help='train days',
)

@click.option(
    '-t',
    '--train_round',
    type=int,
    default=10000,
    help='train round',
)
@click.option(
    '--plot/--no-plot',
    # default=os.name != "nt",
    is_flag=True,
    default=False,
    help="render when training"
)

@click.option(
    '-m',
    '--model_path',
    default='.',
    show_default=True,
    help='trained model save path.',
)
def execute(symbol, begin, end, days, train_round, plot, model_path):
    env = gym.make('trading-v0').env
    env.initialise(symbol=symbol, start=begin, end=end, days=days)

    EPISODES = train_round

    dqn_agent = DQN(env=env)
    simrors = np.zeros(EPISODES)
    mktrors = np.zeros(EPISODES)
    i = 0
    victory = False
    for episode in range(EPISODES):
        if victory == True:
            break;
        done = False
        cur_state = env.reset().reshape(1, env.observation_space.shape[0])
        while not done:
            if episode >= PLOT_AFTER_ROUND:
                #####################################################################################
                if plot:
                    env.render()
                    ####################################################################################
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            new_state = new_state.reshape(1, env.observation_space.shape[0])
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model
            cur_state = new_state
            i += 1

            if done:
                df = env.sim.to_df()
                simrors[episode] = df.bod_nav.values[-1] - 1  # compound returns
                mktrors[episode] = df.mkt_nav.values[-1] - 1
                if episode % 100 == 0:
                    log.info('year #%6d, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f', episode,
                             simrors[episode], mktrors[episode], simrors[episode] - mktrors[episode])
                    if episode > 10:
                        vict = pd.DataFrame({'sim': simrors[episode - 10:episode],
                                             'mkt': mktrors[episode - 10:episode]})
                        vict['net'] = vict.sim - vict.mkt
                        log.info('vict:%f', vict.net.mean())
                        if vict.net.mean() > 0.2:
                            victory = True
                            log.info('Congratulations, Warren Buffet!  You won the trading game ', )
                            break

    import os
    log.info("Completed in %d trials , save it as %s", episode,
             os.path.join(model_path, dqn_agent.env.src.symbol + ".model"))
    dqn_agent.save_model(os.path.join(model_path, dqn_agent.env.src.symbol + ".model"))
    #break

if __name__ == '__main__':
    execute()
