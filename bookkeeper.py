import numpy as np
import pickle
#https://stackoverflow.com/questions/1499119/python-importing-package-classes-into-console-global-namespace
from game_model import *

class Bookkeeper:
    def __init__(self, render, model, prep):
        self.reset()
        self.episode_number = 0
        self.running_reward = None
        self.render = render
        self.model = model
        global preprocess_observation
        preprocess_observation = prep
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
        """
        Note that:
        start = monotonic_ns()
        for i in range(8000000):
            self.xs.append(x.revel())
        print(monotonic_ns() - start)
        Combined with:
        start = monotonic_ns()
        for i in range(8000000):
            self.xs.append(x)
        print(monotonic_ns() - start)
        Indicates ravel accounts for somehting like 200 nanoseconds of the 300 nanoseconds of this append.
        If we assume 10 observations reported per game, and that a game lasts 10^9 nanoseconds,
        this should be gotten to eventually, but is not a high performance priority.
        """
        # Turn our matrices back into vectors so that np.vstack behaves nicely.
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
        # Since the state is a vector which we are treating as a matrix to make life easier,
        # the order does not matter. However we will eventually want to put our x vectors
        # together into a bigger matrix, and we want each column to be an x vector. Therefore
        # we want column-major order for our x vectors.
        self.state = np.zeros((n,1), order = 'F')
        self.state[OFFSET_HEALTH + TEAM_SIZE + 5] = FULL_HEALTH
        self.state[OFFSET_HEALTH + TEAM_SIZE + 4] = FULL_HEALTH
        self.state[OFFSET_HEALTH + TEAM_SIZE + 3] = FULL_HEALTH
        self.state[OFFSET_HEALTH + TEAM_SIZE + 2] = FULL_HEALTH
        self.state[OFFSET_HEALTH + TEAM_SIZE + 1] = FULL_HEALTH
        self.state[OFFSET_HEALTH + TEAM_SIZE] = FULL_HEALTH
        self.state[OFFSET_HEALTH+4] = FULL_HEALTH
        self.state[OFFSET_HEALTH + 1] = FULL_HEALTH

        self.opp_state = np.zeros((n,1), order = 'F')
        self.opp_state[OFFSET_HEALTH + TEAM_SIZE + 5] = FULL_HEALTH
        self.opp_state[OFFSET_HEALTH + TEAM_SIZE + 4] = FULL_HEALTH
        self.opp_state[OFFSET_HEALTH + TEAM_SIZE + 3] = FULL_HEALTH
        self.opp_state[OFFSET_HEALTH + TEAM_SIZE + 2] = FULL_HEALTH
        self.opp_state[OFFSET_HEALTH + TEAM_SIZE + 1] = FULL_HEALTH
        self.opp_state[OFFSET_HEALTH + TEAM_SIZE] = FULL_HEALTH
        self.opp_state[OFFSET_HEALTH+4] = FULL_HEALTH
        self.opp_state[OFFSET_HEALTH + 1] = FULL_HEALTH

        #self.switch_indices = [0,1,2,3,4,5]
        def report_observation(observation):
            state_updates = preprocess_observation(observation)
            for update in state_updates:
                index, value = update
                # check for a new Pokemon switching in. if it did, reset the stat boosts on the relevant side of the field.
                if index < OFFSET_HEALTH:
                    if self.state[index] != value:
                        for i in range(NUM_STAT_BOOSTS):
                            self.state[OFFSET_STAT_BOOSTS + i + NUM_STAT_BOOSTS *(index > NUM_POKEMON)] = 0
                # preprocess_observation returns its absolute stat boosts as integers,
                # while preprocess_observation_smogon returns its relative stat boosts as floats.
                if type(value) == float:
                    assert(index >= OFFSET_STAT_BOOSTS and index < OFFSET_WEATHER)
                    self.state[index] += int(value)
                else:
                    assert(type(value) == int or type(value) == bool)
                    self.state[index] = value
                # Switch around the index so it indexes into the opp_state correctly.
                # TODO: MAKE THIS NICER
                if index < NUM_POKEMON:
                    index += NUM_POKEMON
                elif index < OFFSET_HEALTH:
                    index -= NUM_POKEMON
                elif index < OFFSET_HEALTH + TEAM_SIZE:
                    index += TEAM_SIZE
                elif index < OFFSET_STATUS_CONDITIONS:
                    index -= TEAM_SIZE
                elif index < OFFSET_STATUS_CONDITIONS + TEAM_SIZE * NUM_STATUS_CONDITIONS:
                    index += NUM_STATUS_CONDITIONS
                elif index < OFFSET_STAT_BOOSTS:
                    index -= NUM_STATUS_CONDITIONS
                elif index < OFFSET_STAT_BOOSTS + NUM_STAT_BOOSTS:
                    index += NUM_STAT_BOOSTS
                elif index < OFFSET_WEATHER:
                    index -= NUM_STAT_BOOSTS
                # Do the same thing we just did, except with opp_state.
                if index < OFFSET_HEALTH:
                    if self.opp_state[index] != value:
                        for i in range(NUM_STAT_BOOSTS):
                            self.opp_state[OFFSET_STAT_BOOSTS + i + NUM_STAT_BOOSTS *(index > NUM_POKEMON)] = 0
                if type(value) == float:
                    assert(index >= OFFSET_STAT_BOOSTS and index < OFFSET_WEATHER)
                    self.opp_state[index] += int(value)
                else:
                    assert(type(value) == int or type(value) == bool)
                    self.opp_state[index] = value

            return self.state, self.opp_state
        return report_observation



