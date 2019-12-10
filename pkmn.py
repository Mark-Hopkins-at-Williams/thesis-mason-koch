""" Trains an agent with (stochastic Policy Gradients on Pokemon. Interface inspired by OpenAI Gym."""
import numpy as np
import pickle    # I don't see any particular reason to remove pickle instead of writing to file some other way
from env_pkmn import Env as pkmn_env
from bookkeeper import Bookkeeper

# hyperparameters
from game_model import n    # n used to be in hyperparameters, now it is being imported
from game_model import OUR_TEAM
from game_model import OPPONENT_TEAM
from game_model import POSSIBLE_ACTIONS
H = 64       # number of hidden layer neurons
H2 = 32      # number of hidden layer neurons in second layer
A = 10       # number of actions (one of which, switching to the current pokemon, is always illegal)
batch_size = 2 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False # rendering is so three months from now
np.random.seed(108)

"""The following four functions are vaguely general"""
# it seems inconceivable that sigmoid is not included in numpy, yet it is so
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# relu hidden layer. should be easily swappable with, for instance, sigmoid_hidden_layer.
def relu_hidden_layer(weights, biases, x):
    # Assert that the inputs have the right shape. e.g. shape of (9,) is not allowed.
    assert(len(weights.shape) == 2)
    assert(len(biases.shape) == 2 and biases.shape[1] == 1)
    assert(len(x.shape) == 2 and x.shape[1] == 1)
    retval = weights @ x + biases
    retval[retval<0] = 0
    return retval

# the counterpart of relu_hidden_layer
def backprop_relu_hidden_layer(delta, weights, h):
    # Assert that the inputs have the right shape. (Not checking that, for instance, h and reval have
    # the same shape. That would be pointless extra code).
    assert(len(delta.shape) == 2)
    assert(len(weights.shape) == 2)
    assert(len(h.shape) == 2)
    retval = weights.T @ delta
    retval[h <= 0] = 0
    return retval

# discounting rewards is pretty general. this assumes the game has a reward only at the end,
# which is not unusual.
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

"""Functions like the following three generally exist, but they are different based on game/model"""

def policy_forward(x):
    # Neural network begins here
    h = relu_hidden_layer(model['W1'], model['b1'], x)
    h2 = relu_hidden_layer(model['W2'], model['b2'], h)
    # Neural network ends here.
    # Output layer.
    pvec = np.dot(model['W3'], h2) + model['b3']
    # Softmax. Might be worth putting this into a separate function, might not.
    pvec = np.exp(pvec)
    pvec = pvec / np.sum(pvec)
    # Return the probability vector and the hidden state. 
    # The latter is not strictly necessary, but it will make our lives easier
    return pvec, h, h2

def policy_backward(bookkeeper):
    # Stack all the data from the bookkeeper.
    # This is a fine solution to get the variables into the right shape, but it could be improved
    xs = np.vstack(bookkeeper.xs).T
    hs = np.vstack(bookkeeper.hs).T
    h2s = np.vstack(bookkeeper.h2s).T
    pvecs = np.vstack(bookkeeper.pvecs).T
    actions = np.vstack(bookkeeper.actions) # No point transposing a scalar
    rewards = np.vstack(bookkeeper.rewards)

    # The idea is similar to https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf?fbclid=IwAR2pPF1cbaCMVrdi0qM8lj4xHDDA0uzZem2sjNReUtzdNDKDe7gg5h70sco.
    # We don't know what y is, but we can guess. Also, the naming conventions are different. delta1 and delta2 are switched.
    delta3 = pvecs
    discounted_rewards = discount_rewards(rewards.ravel())
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    for i in range(discounted_rewards.shape[0]):
        delta3[actions[i][0]][i] -= discounted_rewards[i]
    delta2 = backprop_relu_hidden_layer(delta3, model['W3'], h2s)
    dW3 = delta3 @ h2s.T
    db3 = np.sum(delta3,1)
    db3.shape = (db3.shape[0], 1)

    delta1 = backprop_relu_hidden_layer(delta2, model['W2'], hs)
    dW2 = delta2 @ hs.T
    db2 = np.sum(delta2,1)
    db2.shape = (db2.shape[0], 1)

    dW1 = delta1 @ xs.T
    db1 = np.sum(delta1,1)
    db1.shape = (db1.shape[0], 1)

    return {'W1':dW1, 'W2':dW2, 'W3':dW3, 'b1': db1, 'b2':db2, 'b3':db3}

# This is not finalised. in particular, env.seed and env.action_space.seed need to be implemented
def construct_environment():
    env = pkmn_env()
    observation = env.reset()
    return env, observation

# I would be very surprised if this was ever implemented.
def visualize_environment(env):
    if render: 
        env.render()

class RmsProp:
    def __init__(self, model):
        self.grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
    # again, this function is specific to rmsprop.
    def step(self, grad):
        for k in model:
            if k not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']:
                self.grad_buffer[k] += grad[k] # accumulate grad over batch
        if bookkeeper.episode_number % batch_size == 0:
            for k,v in model.items():
                if k not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']:
                    g = self.grad_buffer[k] # gradient
                    self.rmsprop_cache[k] = decay_rate * self.rmsprop_cache[k] + (1 - decay_rate) * g**2
                    model[k] += learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                    self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
 
def choose_action(x, bookkeeper, action_space):
    # This neural network outputs the probabilities of taking each action.
    pvec, h, h2 = policy_forward(x)
    # This assumes, of course, a specific team.
    # Remove illegal actions from our probability vector and then normalise it.
    for i in range(len(POSSIBLE_ACTIONS)):
        pvec[i] *= POSSIBLE_ACTIONS[i] in action_space
    pvec = pvec/np.sum(pvec)
    # Ravel because np.random.choice does not recognise an nx1 matrix as a vector.
    action_index = np.random.choice(range(A), p=pvec.ravel())
    # Report to the bookkeeper the alphabetical index, but return the game index COMMENT IS OUT OF DATE
    bookkeeper.report(x, h, h2, pvec, action_index)
    return POSSIBLE_ACTIONS[action_index]

def run_reinforcement_learning():
    env, observation = construct_environment()
    report_observation = bookkeeper.construct_observation_handler()
    grad_descent = RmsProp(model)
    while True:
        visualize_environment(env)
        x = report_observation(observation)    
        action = choose_action(x, bookkeeper, env.action_space) 
        observation, reward, done, info = env.step(action)
        bookkeeper.report_reward(reward)
        if done: # an episode finished
            print(reward)  # we can plot this over time, and the trend line will tell us how our training is doing
            # Give backprop everything it could conceivably need
            grad = policy_backward(bookkeeper)
            grad_descent.step(grad)
            
            observation = env.reset() # reset env
            report_observation = bookkeeper.construct_observation_handler()
            bookkeeper.signal_episode_completion()
        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            bookkeeper.signal_game_end(reward)
if __name__ == '__main__':
    # model initialization. this will look very different game to game. 
    # personally I would define a numpy array W and access its elements 
    # like W[1] and W[2], but a dictionary is not strictly wrong.
    if resume:
        model = pickle.load(open('save.p', 'rb'))
        # Assert that this model was trained on the same teams we are currently using
        assert(model['0'] == OUR_TEAM[0])
        assert(model['1'] == OUR_TEAM[1])
        assert(model['2'] == OUR_TEAM[2])
        assert(model['3'] == OUR_TEAM[3])
        assert(model['4'] == OUR_TEAM[4])
        assert(model['5'] == OUR_TEAM[5])
        assert(model['6'] == OPPONENT_TEAM[0])
        assert(model['7'] == OPPONENT_TEAM[1])
        assert(model['8'] == OPPONENT_TEAM[2])
        assert(model['9'] == OPPONENT_TEAM[3])
        assert(model['10'] == OPPONENT_TEAM[4])
        assert(model['11'] == OPPONENT_TEAM[5])

    else:
        model = {}
        model['W1'] = 0.1 * np.random.randn(H,n) / np.sqrt(n) # "Xavier" initialization
        model['b1'] = 0.1*np.random.randn(H) / np.sqrt(H)
        model['b1'].shape = (H,1)  # Stop numpy from projecting this vector onto matrices

        model['W2'] = 0.1*np.random.randn(H2,H) / np.sqrt(H2)
        model['b2'] = 0.1*np.random.randn(H2) / np.sqrt(H2)
        model['b2'].shape = (H2,1)

        model['W3'] = 0.1*np.random.randn(A, H2) / np.sqrt(H2)
        model['b3'] = 0.1*np.random.randn(A) / np.sqrt(A)
        model['b3'].shape = (A,1)

        model['0'] = OUR_TEAM[0]
        model['1'] = OUR_TEAM[1]
        model['2'] = OUR_TEAM[2]
        model['3'] = OUR_TEAM[3]
        model['4'] = OUR_TEAM[4]
        model['5'] = OUR_TEAM[5]
        model['6'] = OPPONENT_TEAM[0]
        model['7'] = OPPONENT_TEAM[1]
        model['8'] = OPPONENT_TEAM[2]
        model['9'] = OPPONENT_TEAM[3]
        model['10'] = OPPONENT_TEAM[4]
        model['11'] = OPPONENT_TEAM[5]

    bookkeeper = Bookkeeper(render, model)
        
    run_reinforcement_learning()
