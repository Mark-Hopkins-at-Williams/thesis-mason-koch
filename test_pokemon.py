import unittest
import numpy as np

#from pkmn import sigmoid
#from pkmn import relu_hidden_layer
#from pkmn import policy_forward
import pkmn
import pickle

class TestPong(unittest.TestCase):
    
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

if __name__ == "__main__":
	unittest.main()
