""" Trains an agent with (stochastic Policy Gradients on Pokemon. Interface inspired by OpenAI Gym."""
import numpy as np
import pickle    # I don't see any particular reason to remove pickle instead of writing to file some other way
from env_pkmn import Env as pkmn_env
from bookkeeper import Bookkeeper

# hyperparameters
from game_model import n    # n used to be in hyperparameters, now it is being imported
H = 10       # number of hidden layer neurons
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

    # I looked at previous commits. Andrej used np.outer, so that is what I used in commit "moved relu backprop into its own function".
    # In branch "experimental", I found that this didn't work when I tried to implement multiple layers. This branch was a test branch,
    # so this one had np.outer. (And now it has np.dot!).
    retval = np.dot(delta, weights.T)
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
    # Neural network ends here.
    # Output layer.
    pvec = np.dot(model['W2'].T, h) + model['b2']
    # Softmax. Might be worth putting this into a separate function, might not.
    pvec = np.exp(pvec)
    pvec = pvec / np.sum(pvec)
    # Return the probability vector and the hidden state. 
    # The latter is not strictly necessary, but it will make our lives easier
    return pvec, h

def policy_backward(bookkeeper):
    # Stack all the data from the bookkeeper.
    xs = np.vstack(bookkeeper.xs)
    hs = np.vstack(bookkeeper.hs)
    pvecs = np.vstack(bookkeeper.pvecs)
    actions = np.vstack(bookkeeper.actions)
    rewards = np.vstack(bookkeeper.rewards)

    # The idea is similar to https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf?fbclid=IwAR2pPF1cbaCMVrdi0qM8lj4xHDDA0uzZem2sjNReUtzdNDKDe7gg5h70sco.
    # We don't know what y is, but we can guess. Also, the naming conventions are different. delta1 and delta2 are switched.
    delta2 = pvecs
    discounted_rewards = discount_rewards(rewards.ravel())
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    for i in range(discounted_rewards.shape[0]):
        delta2[i][actions[i]] -= discounted_rewards[i]
    delta1 = backprop_relu_hidden_layer(delta2, model['W2'], hs)
    dW2 = (delta2.T @ hs).T
    db2 = np.sum(delta2.T,1)
    db2.shape = (db2.shape[0], 1)
    dW1 = delta1.T @ xs
    db1 = np.sum(delta1.T,1)
    db1.shape = (db1.shape[0], 1)
    return {'W1':dW1, 'W2':dW2, 'b1': db1, 'b2':db2}

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
            self.grad_buffer[k] += grad[k] # accumulate grad over batch
        if bookkeeper.episode_number % batch_size == 0:
            for k,v in model.items():
                g = self.grad_buffer[k] # gradient
                self.rmsprop_cache[k] = decay_rate * self.rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
 
def choose_action(x, bookkeeper, action_space):
    # This neural network outputs the probabilities of taking each action.
    pvec, h = policy_forward(x)
    # Experimental: Remove this and just rely on the action_space.
    """
    cur_index = 0
    # Aggron, arceus, cacturne, dragonite, druddigon, uxie
    if x[305] == 1:
        cur_index = 0
        assert(x[492] == 0)
        assert(x[331] == 0)
        assert(x[148] == 0)
        assert(x[620] == 0)
        assert(x[479] == 0)
    elif x[492] == 1:
        cur_index = 1
        assert(x[331] == 0)
        assert(x[148] == 0)
        assert(x[620] == 0)
        assert(x[479] == 0)
    elif x[331] == 1:
        cur_index = 2
        assert(x[148] == 0)
        assert(x[620] == 0)
        assert(x[479] == 0)
    elif x[148] == 1:
        cur_index = 3
        assert(x[620] == 0)
        assert(x[479] == 0)
    elif x[620] == 1:
        cur_index = 4
        assert(x[479] == 0)
    else:
        assert(x[479] == 1)
        cur_index = 5
    # Don't switch to the current Pokemon
    pvec[4+cur_index] = 0
    for i in [0,1,2,3,4,5]:
        if x[809*2+i] == 0:
            # Don't switch to a fainted Pokemon
            pvec[4+i]=0
            # If the fainted Pokemon is the active Pokemon, we cannot use moves either
            if i == cur_index:
                pvec[0]=0
                pvec[1]=0
                pvec[2]=0
                pvec[3]=0
    """
    # This assumes, of course, a specific team.
    possible_choices = ["move 1", "move 2", "move 3", "move 4", "switch aggron", "switch arceus", "switch cacturne", "switch dragonite", "switch druddigon", "switch uxie"]
    # Remove illegal actions from our probability vector and then normalise it.
    for i in range(len(possible_choices)):
        pvec[i] *= possible_choices[i] in action_space
    pvec = pvec/np.sum(pvec)
    # Ravel because np.random.choice does not recognise an nx1 matrix as a vector.
    action_index = np.random.choice(range(A), p=pvec.ravel())
    # Report to the bookkeeper the alphabetical index, but return the game index COMMENT IS OUT OF DATE
    bookkeeper.report(x, h, pvec, action_index)
    return possible_choices[action_index]

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
    else:
        model = {}
        model['W1'] = 0.1 * np.random.randn(H,n) / np.sqrt(n) # "Xavier" initialization
        model['b1'] = 0.1*np.random.randn(H) / np.sqrt(H)
        model['b1'].shape = (H,1)  # Stop numpy from projecting this vector onto matrices
        model['W2'] = 0.1*np.random.randn(H,A) / np.sqrt(H)
        model['b2'] = 0.1*np.random.randn(A) / np.sqrt(A)
        model['b2'].shape = (A,1)

    bookkeeper = Bookkeeper(render, model)
        
    run_reinforcement_learning()
