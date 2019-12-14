import numpy as np
from preprocess_observation_smogon import preprocess_observation
import pickle
#https://stackoverflow.com/questions/1499119/python-importing-package-classes-into-console-global-namespace
from game_model import *
# The only difference between this and bookkeeper is that this imports preprocess_observation_smogon, prints out the update, and prints out stat boosts as relative instead of absolute.

class Bookkeeper:
    def __init__(self, render, model):
        self.reset()
        self.episode_number = 0
        self.running_reward = None
        self.render = render
        self.model = model
    def reset(self):
        self.xs,self.hs,self.pvecs,self.actions,self.rewards = [],[],[],[],[]
        self.reward_sum = 0
    def signal_episode_completion(self):
        self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
        if self.render:
            print('resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward))
        self.episode_number += 1
        self.reset()
        if self.episode_number % 3 == 0: pickle.dump(self.model, open('save.p', 'wb'))
    def signal_game_end(self, reward):
        if self.render:
            print(('ep %d: game finished, reward: %f' % (self.episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
    def report(self, x, h, pvec, action):
        # Turn our matrices back into vectors so that np.vstack behaves nicely
        self.xs.append(x.ravel())
        self.hs.append(h.ravel())          # We don't strictly need to remember h
        self.pvecs.append(pvec.ravel())    # or pvecs, but it will make our lives easier
        self.actions.append(action)
    def report_reward(self, reward):
        self.reward_sum += reward      # Recall that we must see the outcome of the action
        self.rewards.append(reward)    # before we write down the reward for taking it
    def construct_observation_handler(self):
        self.state = np.array([0] * n)
        self.state[NUM_POKEMON*2 + TEAM_SIZE + 5] = 100
        self.state[NUM_POKEMON*2 + TEAM_SIZE + 4] = 100
        self.state[NUM_POKEMON*2 + TEAM_SIZE + 3] = 100
        self.state[NUM_POKEMON*2 + TEAM_SIZE + 2] = 100
        self.state[NUM_POKEMON*2 + TEAM_SIZE + 1] = 100
        self.state[NUM_POKEMON*2 + TEAM_SIZE] = 100
        self.state[NUM_POKEMON*2] = 100
        self.state[NUM_POKEMON*2 + 1] = 100
        # Representing the vector as a matrix makes life easier.
        self.state.shape = (n,1)
        self.switch_indices = [0,1,2,3,4,5]
        def report_observation(observation):
            state_updates, self.switch_indices = preprocess_observation(observation)
            for update in state_updates:
                print(update)
                # NOTE TO SELF: RESET ALL STAT BOOSTS WHEN A NEW POKEMON SWITCHES IN
                if update[0] >= NUM_POKEMON*2 + TEAM_SIZE * 2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 and update[1] < NUM_POKEMON*2 + TEAM_SIZE * 2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2:
                    self.state[update[0]] += update[1]
                else:
                    self.state[update[0]] = update[1]
            return self.state
        return report_observation



