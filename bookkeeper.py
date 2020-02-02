import numpy as np
import pickle
from game_model import *

class Bookkeeper:
    def __init__(self, model, prep):
        self.reset()
        self.episode_number = 0
        self.model = model
        # we want to take preprocess_observation as an argument so that we have only one bookkeeper for
        # both env_pkmn and env_pkmn_smogon. but we want it to be similar, syntactically, to importing
        # preprocess_observation, which is why we are using a global variable.
        self.preprocess_observation = prep
    def reset(self):
        self.xs,self.hs,self.h2s,self.pvecs,self.actions,self.rewards=[],[],[],[],[],[]#,self.legal_action_lists = [],[],[],[],[],[],np.zeros(10)
    def signal_episode_completion(self):
        self.episode_number += 1
        self.reset()
        if self.episode_number % 500 == 0: pickle.dump(self.model, open('save.p', 'wb'))
    def report(self, x, h, h2, pvec, action):#,legal_action_list):
        # Turn our matrices back into vectors so that np.vstack behaves nicely.
        self.xs.append(x.ravel())
        self.hs.append(h.ravel())          # We don't strictly need to remember h or h2
        self.h2s.append(h2.ravel())        # or pvecs, but it will make our lives easier
        self.pvecs.append(pvec.ravel())
        self.actions.append(action)
        #legal_action_lists += legal_action_list
    def report_reward(self, reward, took_action):
        if took_action:
            self.rewards.append(reward)
        else:
            # I'm getting a scenario where the first request in a game is a wait request. In this case
            # self.rewards[-1] doesn't exist. My workaround is to say ``well, we don't actually need to
            # add anything unless the reward is nonzero''...
            if reward != 0:
                self.rewards[-1] += reward
    def construct_observation_handler(self):
        FULL_HEALTH = 100
        # Since the state is a vector which we are treating as a matrix to make life easier,
        # the order does not matter. However we will eventually want to put our x vectors
        # together into a bigger matrix, and we want each column to be an x vector. Therefore
        # we want column-major order for our x vectors.
        self.state = np.zeros((N,1), order = 'F')
        self.opp_state = np.zeros((N,1), order = 'F')
        for i in range(6):
            self.state[OFFSET_HEALTH+TEAM_SIZE + i] = FULL_HEALTH
            self.opp_state[OFFSET_HEALTH+TEAM_SIZE + i] = FULL_HEALTH
        self.state[OFFSET_HEALTH + 4] = FULL_HEALTH      # swellow, ledian
        self.state[OFFSET_HEALTH + 1] = FULL_HEALTH
        self.opp_state[OFFSET_HEALTH + 0] = FULL_HEALTH  # aggron, arceus
        self.opp_state[OFFSET_HEALTH + 1] = FULL_HEALTH

        def report_observation(observation):
            state_updates = self.preprocess_observation(observation)
            for update in state_updates:
                index, value = update
                # check for a new Pokemon switching in. if it did, reset the stat boosts on the relevant side of the field.
                if index < OFFSET_HEALTH:
                    if self.state[index] != value:
                        for i in range(NUM_STAT_BOOSTS):
                            self.state[OFFSET_STAT_BOOSTS + i + NUM_STAT_BOOSTS *(index >= NUM_POKEMON)] = 0
                # preprocess_observation returns its absolute stat boosts as integers,
                # while preprocess_observation_smogon returns its relative stat boosts as floats.
                if type(value) == float:
                    assert(index >= OFFSET_STAT_BOOSTS and index < OFFSET_WEATHER)
                    self.state[index] += int(value)
                else:
                    assert(type(value) == int or type(value) == bool)
                    self.state[index] = value
                # Switch around the index so it indexes into the opp_state correctly.
                # You could do this with modular arithmetic... but it's not clear that would be cleaner.
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
                            self.opp_state[OFFSET_STAT_BOOSTS + i + NUM_STAT_BOOSTS *(index >= NUM_POKEMON)] = 0
                if type(value) == float:
                    assert(index >= OFFSET_STAT_BOOSTS and index < OFFSET_WEATHER)
                    self.opp_state[index] += int(value)
                else:
                    assert(type(value) == int or type(value) == bool)
                    self.opp_state[index] = value

            return self.state, self.opp_state
        return report_observation


