import unittest
import numpy as np

import pkmn
import pickle
from preprocess_observation import preprocess_observation as preprocess_observation

class TestPokemon(unittest.TestCase):
    
    def test_sigmoid(self):
        assert pkmn.sigmoid(0) == 0.5
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
        pkmn.model = pickle.load(open('save_stable.p', 'rb'))
        bookkeeper = pkmn.Bookkeeper()
        env, observation = pkmn.construct_environment()
        report_observation = bookkeeper.construct_observation_handler()
        x = report_observation(observation)
        action = pkmn.choose_action(x, bookkeeper)
        # Judgement and shadow claw are garbage; so is switching
        assert(action == "move 1" or action == "move 3")
    
    def test_preprocessing(self):
        # First test: give it a plausible start state. Note that the format of the input to preprocess_observation
        # is not finalised. If I can get the simulator to return a JSON object of the state, we might be looking
        # at two preprocess_observation classes; one for fighting people online (where you can't just muck around
        # in the guts of the game and get it to return what you want) and one for training.
        state_changes, indices = preprocess_observation("|player|p1|Alice||\r\n|player|p2|Bob||\r\n|teamsize|p1|6\r\n|teamsize|p2|6\r\n|gametype|singles\r\n|gen|7\r\n|tier|[Gen 7] Custom Game\r\n|clearpoke\r\n|poke|p1|Ledian, F|item\r\n|poke|p1|Swellow, L10, F|item\r\n|poke|p1|Malamar, L10, M|item\r\n|poke|p1|Houndoom, L10, M|item\r\n|poke|p1|Victreebel, L10, F|item\r\n|poke|p1|Lugia, L10|item\r\n|poke|p2|Arceus-*|item\r\n|poke|p2|Aggron, L10, M|item\r\n|poke|p2|Dragonite, L10, F|item\r\n|poke|p2|Uxie, L10|item\r\n|poke|p2|Cacturne, L10, F|item\r\n|poke|p2|Druddigon, L10, F|item\r\n|teampreview|24\r\n|\r\n|start\r\n|switch|p1a: Ledian|Ledian, F|272/272\r\n|switch|p2a: Arceus|Arceus-Fighting|402/402\r\n|turn|1\r\np2: Arceus\r\np2: Aggron\r\np2: Dragonite\r\np2: Uxie\r\np2: Cacturne\r\np2: Druddigon\r\nDEADBEEF")
        assert(np.all(indices == [2, 1, 5,3,6,4])) # The order is Arceus/Aggron/Dragonite/Uxie/Cacturne/Druddigon.So Aggron is in position 2, Arceus is in position 1, Cacturne is in position 5, Dragonite is in position 3, Druddigon is in position 6, Uxie is in position 4.
        assert([0,1] in state_changes)  # Our opponent has Ledian, which is Pokemon 1, on the field
        assert([1,1] in state_changes)  # Our AI has Arceus on the field, which is coincidentally also Pokemon 1
        assert([3,272] in state_changes)  # Ledian, which is index 3 overall, has 272 health
        assert([9,402] in state_changes)  # Arceus, which is position 9 overall, has 402 health
        # Second test: Give it another plausible start state.
        state_changes, indices = preprocess_observation("|player|p1|Alice||\r\n|player|p2|Bob||\r\n|teamsize|p1|6\r\n|teamsize|p2|6\r\n|gametype|singles\r\n|gen|7\r\n|tier|[Gen 7] Custom Game\r\n|clearpoke\r\n|poke|p1|Ledian, F|item\r\n|poke|p1|Swellow, L10, F|item\r\n|poke|p1|Malamar, L10, M|item\r\n|poke|p1|Houndoom, L10, M|item\r\n|poke|p1|Victreebel, L10, F|item\r\n|poke|p1|Lugia, L10|item\r\n|poke|p2|Arceus-*|item\r\n|poke|p2|Aggron, L10, M|item\r\n|poke|p2|Dragonite, L10, F|item\r\n|poke|p2|Uxie, L10|item\r\n|poke|p2|Cacturne, L10, F|item\r\n|poke|p2|Druddigon, L10, F|item\r\n|teampreview|24\r\n|\r\n|start\r\n|switch|p1a: Swellow|Ledian, F|42/42\r\n|switch|p2a: Arceus|Arceus-Fighting|402/402\r\n|turn|1\r\np2: Arceus\r\np2: Aggron\r\np2: Dragonite\r\np2: Uxie\r\np2: Cacturne\r\np2: Druddigon\r\nDEADBEEF")
        assert(np.all(indices == [2, 1, 5,3,6,4]))
        assert([0,4] in state_changes)  # Now our opponent has Swellow on the field
        assert([1,1] in state_changes)
        assert([6,42] in state_changes)
        assert([9,402] in state_changes)

if __name__ == "__main__":
	unittest.main()
