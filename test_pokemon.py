import unittest
import numpy as np
import json
import pkmn
import pickle
from preprocess_observation import preprocess_observation
from bookkeeper import Bookkeeper
from game_model import *

class TestPokemon(unittest.TestCase):
    
    def test_relu(self):
        # basic tests
        weights = np.array([[2,7,1,8], [6,0,2,2]])
        biases = np.array([1, 3]).T
        biases.shape = (biases.shape[0],1)
        x = np.array([3,1,4,5]).T
        x.shape = (x.shape[0],1)
        assert np.all(pkmn.relu_hidden_layer(weights, biases, x) == [[58],[39]])

        weights = np.array([[2,-4,0,0], [-1,3,3,-7]])
        biases = np.array([0, 0]).T
        biases.shape = (biases.shape[0],1)
        x = np.array([6,9,4,2]).T
        x.shape = (x.shape[0],1)
        assert np.all(pkmn.relu_hidden_layer(weights, biases, x) == [[0],[19]])

    def test_loaded_model(self):
        f = open('save.p', 'rb')
        pkmn.list_of_models, _1, _2 = pickle.load(f)
        f.close()
        bookkeeper = pkmn.Bookkeeper(pkmn.model, preprocess_observation)
        env, observation = pkmn.construct_environment()
        report_observation = bookkeeper.construct_observation_handler()
        x, _ = report_observation(observation)
        action = pkmn.choose_action(x, bookkeeper, env.action_space)
        # Since we don't have a good model anymore, it is OK if this assertion fails
        assert(action == "move 1" or action == "move 3")

    def test_preprocessing_offline(self):
        # First test: give it a plausible start state.
        obs = {
          "active":[
            {"moves":[
              {"move":"bravebird","id":"thunderbolt","pp":24,"maxpp":24,"target":"normal","disabled":False},
              {"move":"earthquake","id":"shadowclaw","pp":24,"maxpp":24,"target":"normal","disabled":False},
              {"move":"swordsdance","id":"aerialace","pp":32,"maxpp":32,"target":"any","disabled":False},
              {"move":"facade","id":"judgment","pp":16,"maxpp":16,"target":"normal","disabled":False}
           ]}
           ],
          "side":
              {"name":"HughMann","id":"p2","pokemon":[
                {"ident":"p2: Swellow","details":"Swellow","condition":"402/402","active":True,"stats":{"atk":245,"def":297,"spa":297,"spd":297,"spe":297},"moves":["bravebird","earthquake","swordsdance","facade"],"baseAbility":"guts","item":"flameorb","pokeball":"pokeball","ability":"guts"},
                {"ident":"p2: Ledian","details":"Ledian","condition":"39/39","active":False,"stats":{"atk":32,"def":46,"spa":22,"spd":22,"spe":20},"moves":["toxic","stealthrock","uturn","roost"],"baseAbility":"swarm","item":"leftovers","pokeball":"pokeball","ability":"swarm"},
              ]},
              "State":
                ["40/40","272/272","46/46","42/42","37/37","41/41",0, "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", '', '', [], []]}
        state_changes, our_active, opponent_active = preprocess_observation(json.dumps(obs))
        assert([809+305,1] in state_changes)  # Our opponent has Aggron, which is Pokemon 306, on the field
        assert([276,1] in state_changes)  # Our AI has Swellow on the field
        assert([OFFSET_HEALTH + TEAM_SIZE + 0,40] in state_changes)  # Aggron, which is index 0 overall, has 40 health
        assert([OFFSET_HEALTH + 4, 402] in state_changes)  # Swellow's health
        assert(our_active == 4)
        assert(opponent_active == 0)
        
    def test_bookkeeper(self):
        f = open('save.p', 'rb')
        pkmn.model, _1, _2 = pickle.load(f)
        f.close()
        from preprocess_observation import preprocess_observation
        bookkeeper = Bookkeeper(pkmn.model, preprocess_observation)
        report_observation = bookkeeper.construct_observation_handler()
        # test report_observation
        report_observation('{"active":[{"moves":[{"move":"Thunderbolt","id":"thunderbolt","pp":24,"maxpp":24,"target":"normal","disabled":false},{"move":"Shadow Claw","id":"shadowclaw","pp":24,"maxpp":24,"target":"normal","disabled":false},{"move":"Aerial Ace","id":"aerialace","pp":32,"maxpp":32,"target":"any","disabled":false},{"move":"Judgment","id":"judgment","pp":16,"maxpp":16,"target":"normal","disabled":false}]}],"side":{"name":"HughMann","id":"p2","pokemon":[{"ident":"p2: Swellow","details":"Swellow","condition":"69/402","active":true,"stats":{"atk":245,"def":297,"spa":297,"spd":297,"spe":297},"moves":["thunderbolt","shadowclaw","aerialace","judgment"],"baseAbility":"guts","item":"flameorb","pokeball":"pokeball","ability":"guts"},{"ident":"p2: Ledian","details":"Ledian, L10, M","condition":"39/39","active":false,"stats":{"atk":32,"def":46,"spa":22,"spd":22,"spe":20},"moves":["roar","heavyslam","rockslide","toxic"],"baseAbility":"swarm","item":"leftovers","pokeball":"pokeball","ability":"sturdy"}]},"State":["40/40","42/272","46/46","42/42","37/37","41/41",0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "", "", [], []]}')
        for i in range(809*2):
            assert(bookkeeper.state[i] == (i == (809+305) or i == 276)  )
        assert(bookkeeper.state[OFFSET_HEALTH + TEAM_SIZE] == 40)
        assert(bookkeeper.state[OFFSET_HEALTH + 4] == 69)
        # test report_reward
        bookkeeper.report_reward(42, True)
        assert(np.all(bookkeeper.rewards == [42]))
        assert(len(bookkeeper.rewards) == 1)
        bookkeeper.report_reward(42, False)
        assert(np.all(bookkeeper.rewards == [84]))
        assert(len(bookkeeper.rewards) == 1)
        # test report
        x = np.zeros((N,1), order = 'F')
        x[276] = 1
        x[809+305] = 1
        FULL_HEALTH = 100
        x[OFFSET_HEALTH + TEAM_SIZE + 5] = FULL_HEALTH
        x[OFFSET_HEALTH + TEAM_SIZE + 4] = FULL_HEALTH
        x[OFFSET_HEALTH + TEAM_SIZE + 3] = FULL_HEALTH
        x[OFFSET_HEALTH + TEAM_SIZE + 2] = FULL_HEALTH
        x[OFFSET_HEALTH + TEAM_SIZE + 1] = FULL_HEALTH
        x[OFFSET_HEALTH + TEAM_SIZE] = FULL_HEALTH
        x[OFFSET_HEALTH] = FULL_HEALTH
        x[OFFSET_HEALTH + 1] = FULL_HEALTH
        x[OFFSET_WEATHER] = 1
        x[OFFSET_TERRAIN] = 1
        f = open('save.p', 'rb')
        pkmn.list_of_models, _1, _2 = pickle.load(f)
        f.close()
        pvec, h, h2 = pkmn.policy_forward(x, pkmn.list_of_models[bookkeeper.our_active][bookkeeper.opponent_active])
        action = 3
        # Report the same data three times, because why not?
        bookkeeper.report(x, h, h2, pvec, action)
        bookkeeper.report(x, h, h2, pvec, action)
        bookkeeper.report(x, h, h2, pvec, action)
        assert(len(bookkeeper.xs) == 3)
        assert(bookkeeper.xs[0].shape == (N,1))
        assert(bookkeeper.xs[1].shape == (N,1))
        assert(bookkeeper.xs[2].shape == (N,1))

        assert(len(bookkeeper.hs) == 3)
        assert(bookkeeper.hs[0].shape == (pkmn.H,1))
        assert(bookkeeper.hs[1].shape == (pkmn.H,1))
        assert(bookkeeper.hs[2].shape == (pkmn.H,1))

        assert(len(bookkeeper.h2s) == 3)
        assert(bookkeeper.h2s[0].shape == (pkmn.H2,1))
        assert(bookkeeper.h2s[1].shape == (pkmn.H2,1))
        assert(bookkeeper.h2s[2].shape == (pkmn.H2,1))

        assert(len(bookkeeper.pvecs) == 3)
        assert(bookkeeper.pvecs[0].shape == (pkmn.A,1))
        assert(bookkeeper.pvecs[1].shape == (pkmn.A,1))
        assert(bookkeeper.pvecs[2].shape == (pkmn.A,1))

        actions = np.vstack(bookkeeper.actions)
        assert(actions.shape == (3,1))
        
if __name__ == "__main__":
	unittest.main()
