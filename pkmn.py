""" Trains an agent with stochastic policy gradients on Pokemon. Interface inspired by OpenAI Gym.
    Some elements of https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5 made their way in here.
"""
import numpy as np
import pickle    # I don't see any particular reason to remove pickle instead of writing to file some other way
import sys
assert(len(sys.argv) <= 2)
if len(sys.argv) == 2:
    assert(sys.argv[1] == "smogon")
    # It drives me nuts that this variables is in the global namespace,
    # yet it is so
    from env_pkmn_smogon import Env as pkmn_env
    from preprocess_observation_smogon import preprocess_observation
else:
    from env_pkmn import Env as pkmn_env
    from preprocess_observation import preprocess_observation
from bookkeeper import Bookkeeper
from game_model import *
# hyperparameters
H = 64       # number of hidden layer neurons
H2 = 32      # number of hidden layer neurons in second layer
A = 10       # number of actions (one of which, switching to the current pokemon, is always illegal)
batch_size = 100 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
np.random.seed(108)

# relu hidden layer. should be easily swappable with, for instance, sigmoid_hidden_layer (not included).
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

# this assumes the game has a reward only at the end, which is not unusual.
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary.
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x, cur_model):
    # Neural network begins here
    h = relu_hidden_layer(cur_model['W1'], cur_model['b1'], x)
    h2 = relu_hidden_layer(cur_model['W2'], cur_model['b2'], h)
    # Neural network ends here.
    # Output layer.
    pvec = np.dot(cur_model['W3'], h2) + cur_model['b3']
    # Softmax. Might be worth putting this into a separate function, might not.
    pvec = np.exp(pvec)
    pvec = pvec / np.sum(pvec)
    # Return the probability vector and the hidden state. 
    # The latter is not strictly necessary, but it will make our lives easier
    return pvec, h, h2

def policy_backward(bookkeeper):
    # Stack all the data from the bookkeeper.
    xs = np.vstack(bookkeeper.xs).T
    hs = np.vstack(bookkeeper.hs).T
    h2s = np.vstack(bookkeeper.h2s).T
    pvecs = np.vstack(bookkeeper.pvecs).T
    actions = np.vstack(bookkeeper.actions)
    rewards = np.vstack(bookkeeper.rewards)
    # Note that all of these arrays are Fortran contiguous:
    assert(xs.flags.f_contiguous)
    assert(hs.flags.f_contiguous)
    assert(h2s.flags.f_contiguous)
    assert(pvecs.flags.f_contiguous)
    # Actions and rewards trivially so:
    assert(actions.flags.f_contiguous)
    assert(actions.flags.c_contiguous)

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

class RmsProp:
    def __init__(self, model):
        self.grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
        self.rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
    # this function is specific to rmsprop.
    def step(self, grad):
        for k in model:
            if k not in range(12):
                self.grad_buffer[k] += grad[k] # accumulate grad over batch
        if bookkeeper.episode_number % batch_size == 0:
            print("Updating weights")
            for k,v in model.items():
                if k not in range(12):
                    g = self.grad_buffer[k] # gradient
                    self.rmsprop_cache[k] = decay_rate * self.rmsprop_cache[k] + (1 - decay_rate) * g**2
                    model[k] -= learning_rate * g / (np.sqrt(self.rmsprop_cache[k]) + 1e-5)
                    self.grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
 
def choose_action(x, bookkeeper, action_space):
    # This neural network outputs the probabilities of taking each action.
    pvec, h, h2 = policy_forward(x, model)
    # This assumes, of course, a specific team.
    # Remove illegal actions from our probability vector and then normalise it.
    if len(sys.argv) == 2:
        HOUNDOOM, LEDIAN, LUGIA, MALAMAR, SWELLOW, VICTREEBEL = 228, 165, 248, 686, 276, 70
        #AGGRON, ARCEUS, CACTURNE, DRAGONITE, DRUDDIGON, UXIE = 305, 492, 331, 148, 620, 479
        # This ifelse statement is far too long, but I can't for the life of me figure out a better way to do it.
        if x[HOUNDOOM] == 1:
            cur_index = 0
            assert(x[LEDIAN] == 0)
            assert(x[LUGIA] == 0)
            assert(x[MALAMAR] == 0)
            assert(x[SWELLOW] == 0)
            assert(x[VICTREEBEL] == 0)
        elif x[LEDIAN] == 1:
            cur_index = 1
            assert(x[LUGIA] == 0)
            assert(x[MALAMAR] == 0)
            assert(x[SWELLOW] == 0)
            assert(x[VICTREEBEL] == 0)
        elif x[LUGIA] == 1:
            cur_index = 2
            assert(x[MALAMAR] == 0)
            assert(x[SWELLOW] == 0)
            assert(x[VICTREEBEL] == 0)
        elif x[MALAMAR] == 1:
            cur_index = 3
            assert(x[SWELLOW] == 0)
            assert(x[VICTREEBEL] == 0)
        elif x[SWELLOW] == 1:
            cur_index = 4
            assert(x[VICTREEBEL] == 0)
        else:
            assert(x[VICTREEBEL] == 1)
            cur_index = 5
        # Don't switch to the current Pokemon
        pvec[4+cur_index] = 0
        for i in range(6):
            if x[809*2+i] == 0:
                # Don't switch to a fainted Pokemon
                pvec[4+i]=0
                # If the fainted Pokemon is the active Pokemon, we cannot use moves either
                if i == cur_index:
                    for j in range(4):
                        pvec[j] = 0
    else:
        for i in range(len(POSSIBLE_ACTIONS)):
            pvec[i] *= POSSIBLE_ACTIONS[i] in action_space
    pvec = pvec/np.sum(pvec)
    # Ravel because np.random.choice does not recognise an nx1 matrix as a vector.
    action_index = np.random.choice(range(A), p=pvec.ravel())
    # Report to the bookkeeper the alphabetical index, but return the game index COMMENT IS OUT OF DATE
    bookkeeper.report(x, h, h2, pvec, action_index)
    return POSSIBLE_ACTIONS[action_index]

def opponent_choose_action(x, bookkeeper, action_space):
    # like choose_action, except we use a different model, cite different constants,
    # and don't report anything to the bookkeeper
    pvec, h, h2 = policy_forward(x, opponent_model)
    for i in range(len(OPPONENT_POSSIBLE_ACTIONS)):
        pvec[i] *= OPPONENT_POSSIBLE_ACTIONS[i] in action_space
    pvec = pvec/np.sum(pvec)
    action_index = np.random.choice(range(A), p=pvec.ravel())
    return OPPONENT_POSSIBLE_ACTIONS[action_index]

def run_reinforcement_learning():
    env, observation = construct_environment()
    report_observation = bookkeeper.construct_observation_handler()
    grad_descent = RmsProp(model)
    while True:
        if len(sys.argv) == 2:
            x, _ = report_observation(observation)
            action = choose_action(x, bookkeeper, env.action_space)
            observation, reward, done, info = env.step(action)
        else:
            assert(len(env.action_space) + len(env.opponent_action_space) > 0)
            x, opp_x = report_observation(observation)
            if len(env.action_space) > 0:
                # Our AI needs to choose a move
                action = choose_action(x, bookkeeper, env.action_space)
            else:
                action = ''
            if len(env.opponent_action_space) > 0:
                # Our opponent needs to make a move
                opponent_action = opponent_choose_action(opp_x, bookkeeper, env.opponent_action_space)
            else:
                opponent_action = ''
            #lenenv = len(env.action_space) # In my heart of hearts, I strongly believe
            # this is a bugfix. However, experimentation has not confirmed this yet.
            observation, reward, done, info = env.step(opponent_action + "|" + action)
            bookkeeper.report_reward(reward, len(env.action_space) > 0)#lenenv > 0)
        if done: # an episode finished
            if len(sys.argv) == 2:
                break
            print(reward, end = " ")  # we can plot this over time, and the trend line will tell us how our training is doing
            # Give backprop everything it could conceivably need
            grad = policy_backward(bookkeeper)
            grad_descent.step(grad)
            observation = env.reset() # reset env
            report_observation = bookkeeper.construct_observation_handler()
            bookkeeper.signal_episode_completion()
if __name__ == '__main__':
    # model initialization. this will look very different game to game. 
    # personally I would define a numpy array W and access its elements 
    # like W[1] and W[2], but a dictionary is not strictly wrong.
    if resume:
        model = pickle.load(open('save.p', 'rb'))
        opponent_model = pickle.load(open('save_opponent.p', 'rb'))
        # check for loaded file compatibility
        for i in range(6):
            assert(model[i] == opponent_model[i+6])
            assert(model[i+6] == opponent_model[i])
        # Assert that the loaded models were trained on the same teams we are currently using
        for i in range(6):
            assert(model[i] == OUR_TEAM[i])
            assert(model[i+6] == OPPONENT_TEAM[i])
    else:
        model = {}
        model['W1'] = 0.1 * np.random.randn(H,N) / np.sqrt(N) # "Xavier" initialization
        model['b1'] = 0.1*np.random.randn(H) / np.sqrt(H)
        model['b1'].shape = (H,1)  # Stop numpy from projecting this vector onto matrices
        model['W2'] = 0.1*np.random.randn(H2,H) / np.sqrt(H2)
        model['b2'] = 0.1*np.random.randn(H2) / np.sqrt(H2)
        model['b2'].shape = (H2,1)
        model['W3'] = 0.1*np.random.randn(A, H2) / np.sqrt(H2)
        model['b3'] = 0.1*np.random.randn(A) / np.sqrt(A)
        model['b3'].shape = (A,1)
        for i in range(6):
            model[i] = OUR_TEAM[i]
            model[i+6] = OPPONENT_TEAM[i]
        opponent_model = {}
        opponent_model['W1'] = 0.1 * np.random.randn(H,N) / np.sqrt(N)
        opponent_model['b1'] = 0.1*np.random.randn(H) / np.sqrt(H)
        opponent_model['b1'].shape = (H,1)
        opponent_model['W2'] = 0.1*np.random.randn(H2,H) / np.sqrt(H2)
        opponent_model['b2'] = 0.1*np.random.randn(H2) / np.sqrt(H2)
        opponent_model['b2'].shape = (H2,1)
        opponent_model['W3'] = 0.1*np.random.randn(A, H2) / np.sqrt(H2)
        opponent_model['b3'] = 0.1*np.random.randn(A) / np.sqrt(A)
        opponent_model['b3'].shape = (A,1)
        for i in range(6):
            opponent_model[i] = OPPONENT_TEAM[i]
            opponent_model[i+6] = OUR_TEAM[i]
        pickle.dump(model, open('save.p', 'wb'))
        pickle.dump(opponent_model, open('save_opponent.p', 'wb'))


    bookkeeper = Bookkeeper(model, preprocess_observation)
    run_reinforcement_learning()
