""" Trains an agent with (stochastic Policy Gradients on Pokemon. Interface inspired by OpenAI Gym."""
import numpy as np
import pickle    # I don't see any particular reason to remove pickle instead of writing to file some other way
#import gym    # We are not using gym anymore, but I'm not going to flat-out delete it quite yet
from pkmn_env import Env as pkmn_env

# hyperparameters
H = 200 # number of hidden layer neurons
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
def relu_hidden_layer(weights, x):
    retval = weights @ x
    retval[retval<0] = 0
    return retval

# the counterpart of relu_hidden_layer
def backprop_relu_hidden_layer(delta, weights, h):
    retval = np.outer(delta, weights)
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

"""Functions like the following four generally exist, but they are different based on game/model"""

def preprocess_observation(I):
    # There will always be a preprocessing step, but it will look different for different games.
    # Return a list. Each element of the list contains two elements, the index of the state to update
    # and the value to update it to.
    return np.array([[0,42]])

def policy_forward(x):
    # This will need to be overhauled in the future
    return 42
    # Neural network begins here
    #h = relu_hidden_layer(model['W1'], x)
    # Neural network ends here. In a later commit, we will add multiple hidden layers to show it can be done.
    # Output layer. This is going to look largely the same if we are wanting two probabilities as our output.
    #logitp = np.dot(model['W2'], h)
    #p = sigmoid(logitp)
    # Return the number we are interested in (in this case the probability of taking action 2) and the output
    # the hidden states. The latter is not strictly necessary, but it will make our lives easier
    return p, h

def policy_backward(bookkeeper):
    # stack together all inputs, hidden states, probabilities, actions and rewards for this episode
    xs = np.vstack(bookkeeper.xs)
    hs = np.vstack(bookkeeper.hs)                # Both h and prob_action_2 are strictly functions of x, so we don't need to
    prob_action_2s = np.vstack(bookkeeper.prob_action_2s)    # remember them. But we should, because that will be less work.
    actions = np.vstack(bookkeeper.actions)
    rewards = np.vstack(bookkeeper.rewards)
 
    # Implement Andrej's vaguely strange gradient
    actions = (actions - 1) % 2  # 1 if action is 2, 0 otherwise 
    actions = actions.astype('float64') - prob_action_2s
    
    discounted_rewards = discount_rewards(rewards)
    # Standardize the rewards to be unit normal because Andrej says so.
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    delta2 = actions * discounted_rewards
    # Right, now we have this strange quantity.
    dW2 = (hs.T @ delta2).ravel()
    # Do the next layer.
    delta1 = backprop_relu_hidden_layer(delta2, model['W2'], hs)
    dW1 = delta1.T @ xs
    return {'W1':dW1, 'W2':dW2}

# This is not finalised. in particular, env.seed and env.action_space.seed need to be implemented
def construct_environment():
    env = pkmn_env()
    observation = env.reset()
    return env, observation

# I would be very surprised if this was ever implemented.
def visualize_environment(env):
    if render: 
        env.render()

class Bookkeeper:
    def __init__(self):
        self.reset()
        self.episode_number = 0
        self.running_reward = None
    def reset(self):
        self.xs,self.hs,self.prob_action_2s,self.actions,self.rewards = [],[],[],[],[]
        self.reward_sum = 0
    def signal_episode_completion(self):
        self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
        if render: 
            print('resetting env. episode reward total was %f. running mean: %f' % (self.reward_sum, self.running_reward))
        self.episode_number += 1
        self.reset()
        if self.episode_number % 3 == 0: pickle.dump(model, open('save.p', 'wb'))
    def signal_game_end(self, reward):
        if render:
            print(('ep %d: game finished, reward: %f' % (self.episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
    def report(self, x, h, prob_action_2, action):
        self.xs.append(x)
        self.hs.append(h)    # We don't strictly need to remember this, 
                             # but it will make our lives easier
        self.prob_action_2s.append(prob_action_2)    # Same
        self.actions.append(action)
    def report_reward(self, reward):
        self.reward_sum += reward
        self.rewards.append(reward)    # Recall that we must see the outcome 
                                       # of the action before we write down
                                       # the reward for taking it
    def construct_observation_handler():
        # First, let the observation be the health of both team's Pokemon
        self.state = np.array([100, 100, 100, 100, 100, 100,   100, 100, 100, 100, 100, 100])
        def report_observation(observation):
            # This function seemed vaguely necessary for pong, where the change in the
            # observation was important, but for Pokemon methinks it is a glorified wrapper
            new_information = preprocess_observation(observation)
            for info in new_information:
                self.state[new_information[0]] = new_information[1]
            return self.state
        return report_observation


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
 
def choose_action(x):
    # This neural network outputs the probability of taking action 2. 
    # This is unusual, normally it would output the probabilities for taking
    # each action. But, if we have two actions, it's not wrong.

    #prob_action_2, h = policy_forward(x)
    #action = 2 if np.random.uniform() < prob_action_2 else 3    
    h = 3.14
    prob_action_2 = 2.78
    action = "move 1"
    bookkeeper.report(x, h, prob_action_2, action)
    return action

def run_reinforcement_learning():
    env, observation = construct_environment()
    report_observation = bookkeeper.construct_observation_handler()
    grad_descent = RmsProp(model)
    while True:
        visualize_environment(env)
        x = report_observation(observation)    
        action = choose_action(x)  
        print(observation)
        print(".")
        observation, reward, done, info = env.step(action)
        bookkeeper.report_reward(reward)
        print(observation)
        print(".")
        print(crash)  # Everything beyond this we don't need to worry about yet
        if done: # an episode finished
            # Give backprop everything it could conceivably need
            grad = policy_backward(bookkeeper)
            grad_descent.step(grad)
            
            observation = env.reset() # reset env
            report_observation = construct_observation_handler()
            bookkeeper.signal_episode_completion()
                  
        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            bookkeeper.signal_game_end(reward)
    

if __name__ == '__main__':
    # model initialization. this will look very different game to game. 
    # personally I would define a numpy array W and access its elements 
    # like W[1] and W[2], but a dictionary is not strictly wrong.
    D = 80 * 80 # input dimensionality: 80x80 grid
    if resume:
        model = pickle.load(open('save.p', 'rb'))
    else:
        model = {}
        model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
        model['W2'] = np.random.randn(H) / np.sqrt(H)
        
    bookkeeper = Bookkeeper()
        
    run_reinforcement_learning()
