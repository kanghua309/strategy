# -*- coding: utf-8 -*-

import logging

import click
import gym
import gym_trading  #必须引入才自动注册

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

PLOT_AFTER_ROUND = 1

log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)
# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.hidden1, self.hidden2 = 24, 24

        # create model for policy network
        self.model = self.build_model()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        # if self.load_model:
        #    self.model.load_weights("./save_model/cartpole_reinforce.h5")

    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()
        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a).
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        return model

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)

        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

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
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('trading-v0').env
    env.initialise(symbol=symbol, start=begin, end=end, days=days)
    EPISODES = train_round
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes = [], []
    simrors = np.zeros(EPISODES)
    mktrors = np.zeros(EPISODES)
    victory = False
    for episode in range(EPISODES):
        if victory:
            break
        done = False
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if episode >= PLOT_AFTER_ROUND:
                #####################################################################################
                if plot:
                    env.render()
                ####################################################################################
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            #reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r> to the memory
            agent.append_sample(state, action, reward)

            state = next_state

            if done:
                df = env.sim.to_df()
                simrors[episode] = df.bod_nav.values[-1] - 1  # compound returns
                mktrors[episode] = df.mkt_nav.values[-1] - 1
                if episode % 100 == 0:
                    log.info('year #%6d, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f', episode,
                             simrors[episode], mktrors[episode], simrors[episode] - mktrors[episode])
                    if episode > 100:
                        vict = pd.DataFrame({'sim': simrors[episode - 100:episode],
                                             'mkt': mktrors[episode - 100:episode]})
                        vict['net'] = vict.sim - vict.mkt
                        log.info('vict:%f', vict.net.mean())
                        if vict.net.mean() > 0.2:
                            victory = True
                            log.info('Congratulations, Warren Buffet!  You won the trading game ', )
                            break

                # every episode, agent learns from sample returns
                agent.train_model()
                episodes.append(episode)

    import os
    log.info("Completed in %d trials , save it as %s", episode,
             os.path.join(model_path, env.src.symbol + ".model"))
    agent.save_model(os.path.join(model_path, env.src.symbol + ".model"))


if __name__ == "__main__":
    execute()