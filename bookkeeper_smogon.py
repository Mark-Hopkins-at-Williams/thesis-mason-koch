import numpy as np
import pickle
#https://stackoverflow.com/questions/1499119/python-importing-package-classes-into-console-global-namespace
from game_model import *


class Bookkeeper:
    def __init__(self, render, model, preprocess_observation):
        self.reset()
        self.episode_number = 0
        self.running_reward = None
        self.render = render
        self.model = model
        self.preprocess_observation = preprocess_observation
    def reset(self):
        self.xs,self.hs,self.h2s,self.pvecs,self.actions,self.rewards = [],[],[],[],[],[]
        self.reward_sum = 0
    def signal_episode_completion(self):
        self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
        if self.render:
            print('resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward))
        self.episode_number += 1
        self.reset()
        if self.episode_number % 500 == 0: pickle.dump(self.model, open('save.p', 'wb'))
    def signal_game_end(self, reward):
        if self.render:
            print(('ep %d: game finished, reward: %f' % (self.episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
    def report(self, x, h, h2, pvec, action):
        # Turn our matrices back into vectors so that np.vstack behaves nicely
        self.xs.append(x.ravel())
        self.hs.append(h.ravel())          # We don't strictly need to remember h or h2
        self.h2s.append(h2.ravel())        # or pvecs, but it will make our lives easier
        self.pvecs.append(pvec.ravel())
        self.actions.append(action)
    def report_reward(self, reward, took_action):
        self.reward_sum += reward      # Recall that we must see the outcome of the action
        if took_action:
            self.rewards.append(reward)    # before we write down the reward for taking it
        else:
            self.rewards[-1] += reward
    def construct_observation_handler(self):
        FULL_HEALTH = 100
        self.state = np.array([0] * n)
        self.state[NUM_POKEMON*2 + TEAM_SIZE + 5] = FULL_HEALTH
        self.state[NUM_POKEMON*2 + TEAM_SIZE + 4] = FULL_HEALTH
        self.state[NUM_POKEMON*2 + TEAM_SIZE + 3] = FULL_HEALTH
        self.state[NUM_POKEMON*2 + TEAM_SIZE + 2] = FULL_HEALTH
        self.state[NUM_POKEMON*2 + TEAM_SIZE + 1] = FULL_HEALTH
        self.state[NUM_POKEMON*2 + TEAM_SIZE] = FULL_HEALTH
        self.state[NUM_POKEMON*2] = FULL_HEALTH
        self.state[NUM_POKEMON*2 + 1] = FULL_HEALTH
        # Representing the vector as a matrix makes life easier.
        self.state.shape = (n,1)

        self.opp_state = np.array([0] * n)
        self.opp_state[NUM_POKEMON*2 + TEAM_SIZE + 5] = FULL_HEALTH
        self.opp_state[NUM_POKEMON*2 + TEAM_SIZE + 4] = FULL_HEALTH
        self.opp_state[NUM_POKEMON*2 + TEAM_SIZE + 3] = FULL_HEALTH
        self.opp_state[NUM_POKEMON*2 + TEAM_SIZE + 2] = FULL_HEALTH
        self.opp_state[NUM_POKEMON*2 + TEAM_SIZE + 1] = FULL_HEALTH
        self.opp_state[NUM_POKEMON*2 + TEAM_SIZE] = FULL_HEALTH
        self.opp_state[NUM_POKEMON*2] = FULL_HEALTH
        self.opp_state[NUM_POKEMON*2 + 1] = FULL_HEALTH
        # Representing the vector as a matrix makes life easier.
        self.opp_state.shape = (n,1)


        self.switch_indices = [0,1,2,3,4,5]
        def report_observation(observation):
            state_updates, self.switch_indices = self.preprocess_observation(observation)
            for update in state_updates:
                index, value = update
                if index >= NUM_POKEMON*2 + TEAM_SIZE * 2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 and index < NUM_POKEMON*2 + TEAM_SIZE * 2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2:
                    # NOTE TO SELF: RESET ALL STAT BOOSTS WHEN A NEW POKEMON SWITCHES IN
                    self.state[index] += value
                else:
                    self.state[index] = value
                # TODO: MAKE THIS NICER
                if index < NUM_POKEMON:
                    index += NUM_POKEMON
                elif index < 2 * NUM_POKEMON:
                    index -= NUM_POKEMON
                elif index < NUM_POKEMON*2 + TEAM_SIZE:
                    index += TEAM_SIZE
                elif index < NUM_POKEMON*2 + TEAM_SIZE*2:
                    index -= TEAM_SIZE
                elif index < NUM_POKEMON*2 + TEAM_SIZE*2 + TEAM_SIZE * NUM_STATUS_CONDITIONS:
                    index += NUM_STATUS_CONDITIONS
                elif index < NUM_POKEMON*2 + TEAM_SIZE*2 + TEAM_SIZE * NUM_STATUS_CONDITIONS*2:
                    index -= NUM_STATUS_CONDITIONS
                elif index < NUM_POKEMON*2 + TEAM_SIZE*2 + TEAM_SIZE * NUM_STATUS_CONDITIONS*2 + NUM_STAT_BOOSTS:
                    index += NUM_STAT_BOOSTS
                elif index < NUM_POKEMON*2 + TEAM_SIZE*2 + TEAM_SIZE * NUM_STATUS_CONDITIONS*2 + NUM_STAT_BOOSTS*2:
                    index -= NUM_STAT_BOOSTS
                if index >= NUM_POKEMON*2 + TEAM_SIZE * 2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 and index < NUM_POKEMON*2 + TEAM_SIZE * 2 + NUM_STATUS_CONDITIONS*TEAM_SIZE*2 + NUM_STAT_BOOSTS*2:
                    self.opp_state[index] += value
                else:
                    self.opp_state[index] = value

            return self.state, self.opp_state
        return report_observation



