"""
Trains an agent with (stochastic Policy Gradients on Pokemon. Interface inspired by OpenAI Gym.
Differences between this and pkmn.py:
This does not use a random seed because the modus operandi for this file is to run it multiple times. Since pmariglia's AI behaves similarly on similar runs, keeping the same random seed would make each battle against pmariglia's AI the same, which would bork our assumptions about statistical significance.
This prints the pvecs and the observations.
This attempts to figure out what actions are legal instead of leaving that to the Pokemon server. (Because we can get the Pokemon server to return what we want if it is running locally on our computer, but if it is running on Pokemon Showdown, that's not an option).
This has removed the code which is not necessary for it, like backpropagation and opponent_choose_action
"""
import numpy as np
import pickle    # I don't see any particular reason to remove pickle instead of writing to file some other way
from env_pkmn_smogon import Env as pkmn_env
from bookkeeper_smogon import Bookkeeper

# hyperparameters
from game_model import n    # n used to be in hyperparameters, now it is being imported
from game_model import OUR_TEAM
from game_model import OPPONENT_TEAM
from game_model import POSSIBLE_ACTIONS
from game_model import OPPONENT_POSSIBLE_ACTIONS
H = 64       # number of hidden layer neurons
H2 = 32      # number of hidden layer neurons in second layer
A = 10       # number of actions (one of which, switching to the current pokemon, is always illegal)
resume = True  # resume from previous checkpoint. this should always be true in _smogon.
render = False # rendering is so three months from now

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

# This is not finalised. in particular, env.seed and env.action_space.seed need to be implemented
def construct_environment():
    env = pkmn_env()
    observation = env.reset()
    return env, observation

# I would be very surprised if this was ever implemented.
def visualize_environment(env):
    if render: 
        env.render()

def choose_action(x):
    # This neural network outputs the probabilities of taking each action.
    pvec, h, h2 = policy_forward(x, model)
    print(pvec)

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

    pvec = pvec/np.sum(pvec)
    print(pvec)
    # Ravel because np.random.choice does not recognise an nx1 matrix as a vector.
    action_index = np.random.choice(range(A), p=pvec.ravel())
    # Up until now, we have been denoting a Pokemon by its alphabetical index.
    # This is not how the Pokemon simulator works. Instead it stores them in some arbitrary order.
    # 0th entry of the switch index is Aggron's position in the arbitrary ordering.
    return POSSIBLE_ACTIONS[action_index]

def run_reinforcement_learning():
    env, observation = construct_environment()
    report_observation = bookkeeper.construct_observation_handler()
    done = False
    while not done:
        visualize_environment(env)
        print(observation)
        x = report_observation(observation)    
        action = choose_action(x) 
        observation, reward, done, info = env.step(action)

if __name__ == '__main__':
    # model initialization. this will look very different game to game. 
    # personally I would define a numpy array W and access its elements 
    # like W[1] and W[2], but a dictionary is not strictly wrong.
    assert(resume)
    model = pickle.load(open('save.p', 'rb'))
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
    bookkeeper = Bookkeeper(render, model)
    run_reinforcement_learning()



