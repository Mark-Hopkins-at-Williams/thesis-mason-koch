import unittest
import numpy as np

from pg_pong import sigmoid
from pg_pong import relu_hidden_layer
from pg_pong import policy_forward
from pg_pong import model

class TestPong(unittest.TestCase):
    
    def test_sigmoid(self):
        assert sigmoid(0) == 0.5
    def test_relu(self):
        # basic tests
        assert np.all(relu_hidden_layer(np.array([[2,7,1,8], [6,0,2,2]]), np.array([3,1,4,5])) == [57,36])
        assert np.all(relu_hidden_layer(np.array([[2,-4,0,0], [-1,3,3,-7]]), np.array([6,9,4,2])) == [0,19])
        # make sure it works just as well as policy_forward does
        x = np.random.randint(2, size=6400)
        h = relu_hidden_layer(model['W1'], x)
        logp = np.dot(model['W2'], h)
        p = sigmoid(logp)
        assert policy_forward(x)[0] == p
        assert np.all(policy_forward(x)[1] == h)

if __name__ == "__main__":
	unittest.main()
