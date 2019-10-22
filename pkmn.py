""" Trains an agent with (stochastic Policy Gradients on Pokemon. Interface inspired by OpenAI Gym."""
import numpy as np
import pickle    # I don't see any particular reason to remove pickle instead of writing to file some other way
#import gym    # We are not using gym anymore, but I'm not going to flat-out delete it quite yet
from env_pkmn import Env as pkmn_env

# hyperparameters
n = 14 # dimensionality of input 
H = 10 # number of hidden layer neurons
A = 9 # number of actions
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
    retval = weights @ x + biases
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
    # In this case, the string we get back from the Pokemon simulator does not give us the entire state
    # of the game. Instead it gives us the change in the state. So return a list.
    # Each element of the list contains two elements, the index of the state to update
    # and the value to update it to.
    retval = []
    I = I.split('\n')
    dbi = -1
    ci = 0
    retval2 = [-1,-1,-1,-1,-1,-1]
    for line in I:
        if ('switch|' in line) or ('drag|' in line):
            print('switch line is ' + line)
            # There is a new Pokemon on the field. Update the pokemon on field and the health.
            if 'p1a' in line:
                temp = line.split('|')
                # Todo in maybe December: Make this handle all the Pokemon and not just our favorites
                tempDict = {'Houndoom': 0, 'Ledian': 1, 'Lugia': 2, 'Malamar': 3, 'Swellow': 4, 'Victreebel': 5}
                name = temp[2][5:]
                index = tempDict[name]
                retval.append([0, index])
                health = int(temp[-1].split('/')[0])
                retval.append([2 + index, health])
            else:
                assert('p2a' in line)
                temp = line.split('|')
                tempDict = {'Aggron': 0, 'Arceus': 1, 'Cacturne': 2, 'Dragonite': 3, 'Druddigon': 4, 'Uxie': 5}
                name = temp[2][5:]
                index = tempDict[name]
                retval.append([1, index])
                health = int(temp[-1].split('/')[0])
                retval.append([8 + index, health])
        elif 'damage' in line:
            if 'Substitute' not in line:
                print('damage line is ' + line)
                tempDict = {'Houndoom': 2, 'Ledian': 3, 'Lugia': 4, 'Malamar': 5, 'Swellow': 6, 'Victreebel': 7, 'Aggron': 8, 'Arceus': 9, 'Cacturne': 10, 'Dragonite': 11, 'Druddigon': 12, 'Uxie': 13}
                temp = line.split('|')
                name = temp[2][5:]
                if temp[-1][0] == '[':
                    #The simulator is telling us the source of the damage
                    health = 0
                    if 'fnt' not in temp[-2]:
                        health = int(temp[-2].split('/')[0])
                    retval.append([tempDict[name], health])
                else:
                    if 'fnt' in temp[-1]:
                        health = 0
                        retval.append([tempDict[name], health])
                    else:
                        health = int(temp[-1].split('/')[0])
                        retval.append([tempDict[name], health])
        elif 'DEADBEEF' in line:
            dbi = ci
        elif line == 'p2: Aggron\r':
            retval2[0] = ci
        elif line == 'p2: Arceus\r':
            retval2[1] = ci
        elif line == 'p2: Cacturne\r':
            retval2[2] = ci
        elif line == 'p2: Dragonite\r':
            retval2[3] = ci
        elif line == 'p2: Druddigon\r':
            retval2[4] = ci
        elif line == 'p2: Uxie\r':
            retval2[5] = ci
        ci += 1

        #There are way, way more parameters we can and should extract from this, but that's what we are doing for now
    retval2 -= np.min(retval2)
    retval2 += 1
    return retval, retval2

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
    def construct_observation_handler(self):
        # First, let the observation be the health of both team's Pokemon and also which Pokemon is active.
        self.state = np.array([-1, -1, 100, 100, 100, 100, 100, 100,   100, 100, 100, 100, 100, 100])
        self.state.shape = (14,1)
        self.switch_indices = [0,1,2,3,4,5]
        def report_observation(observation):
            new_information, self.switch_indices = preprocess_observation(observation)
            for info in new_information:
                self.state[info[0]] = info[1]
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
 
def choose_action(x, switch_indices):
    # This neural network outputs the probabilities of taking each action.
    pvec, h = policy_forward(x)
    # TODO in the far future: remember which Pokemon are fainted, don't recompute every time
    # Don't switch to a fainted pokemon
    for i in [0,1,2,3,4,5]:
        # A Pokemon with index lower than active pokemon
        if i < x[1]:
            if x[8+i] == 0:
                pvec[4+i] = 0
        if i == x[1]:
            if x[8+i] == 0:
                # Current pokemon has fainted, we cannot use moves
                pvec[0]=0
                pvec[1]=0
                pvec[2]=0
                pvec[3]=0
        # A pokemon with index higher than active pokemon
        if i > x[1]:
            if x[8+i] == 0:
                pvec[3+i] = 0

    print(x)
    pvec = pvec/np.sum(pvec)
    print(pvec)
    # Ravel because np.random.choice does not recognise an nx1 matrix as a vector.
    possible_choices = ["move 1", "move 2", "move 3", "move 4", "switch 0", "switch 1", "switch 2", "switch 3", "switch 4",]
    action = np.random.choice(possible_choices, p=pvec.ravel())
    # Up until now, we have been denoting a Pokemon by its alphabetical index.
    # This is not how the Pokemon simulator works. Instead it stores them in some arbitrary order.
    # 0th entry of the switch index is Aggron's position in the arbitrary ordering.
    retval = action
    if 'switch' in action:
        official_index = int(action[-1])
        if official_index >= x[1]:
            official_index += 1
        retval = 'switch ' + str(switch_indices[official_index])
    # Report to the bookkeeper the alphabetical index, but return the game index
    bookkeeper.report(x, h, pvec, action)
    print("Coming out of choose_action, it sure looks like the action is " + retval)
    return retval

def run_reinforcement_learning():
    env, observation = construct_environment()
    report_observation = bookkeeper.construct_observation_handler()
    grad_descent = RmsProp(model)
    while True:
        visualize_environment(env)
        x = report_observation(observation)    
        action = choose_action(x, bookkeeper.switch_indices) 
        observation, reward, done, info = env.step(action)
        bookkeeper.report_reward(reward)
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

    bookkeeper = Bookkeeper()
        
    run_reinforcement_learning()
