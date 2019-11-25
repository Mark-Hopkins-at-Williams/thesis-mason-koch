import numpy as np
from preprocess_observation_smogon import preprocess_observation
import pickle


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
        # First, let the observation be the health of both team's Pokemon and also which Pokemon is active.
        self.state = np.array([0] * 1630)
        self.shape[-1] = 100
        self.shape[-2] = 100
        self.shape[-3] = 100
        self.shape[-4] = 100
        self.shape[-5] = 100
        self.shape[-6] = 100
        self.shape[-11] = 100
        # Representing the vector as a matrix makes life easier.
        self.state.shape = (809*2+12,1)
        self.switch_indices = [0,1,2,3,4,5]
        def report_observation(observation):
            state_updates, self.switch_indices = preprocess_observation(observation)
            for update in state_updates:
                print(update)
                self.state[update[0]] = update[1]
            return self.state
        return report_observation



