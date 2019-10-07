import unittest
import numpy as np

from pg_pong import sigmoid
from pg_pong import relu_hidden_layer
from pg_pong import policy_forward
from pg_pong import model
from pg_pong import sigmoid_hidden_layer
from pg_pong import backprop_sigmoid_hidden_layer


class TestPong(unittest.TestCase):
    
    def test_sigmoid(self):
        x = np.random.randint(2, size=6400)
        x.shape = (6400, 1)
        forward = sigmoid_hidden_layer(model['W1'], x)
        assert(forward.shape == (200,1))
        x = np.zeros([6400,1])
        forward = sigmoid_hidden_layer(model['W1'], x)
        # Doesn't matter what the weights are if x is all zeros, now does it!
        assert(np.all(forward == 0.5))
        five = np.zeros([1,1])
        five += 5
        h = np.zeros([200,1])
        assert(np.all(backprop_sigmoid_hidden_layer(five, model['W3'], h) == 1.25 * model['W3']))
        assert(backprop_sigmoid_hidden_layer(five, model['W3'], h).shape == (200,1))

if __name__ == "__main__":
	unittest.main()
