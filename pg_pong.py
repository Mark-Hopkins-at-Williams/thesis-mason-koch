""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
    Based on http://karpathy.github.io/2016/05/31/rl/."""
import numpy as np
import pickle
import gym    # For full generality, we might not depend on OpenAI Gym. This is comfortably on the back burner.

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 2 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
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
  retval = np.dot(weights, delta.T)
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
# model initialization. this will look very different game to game. personally I would define a numpy array W and access its elements like W[1] and W[2],
# but a dictionary is not strictly wrong.
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H,H) / np.sqrt(H)
  model['W3'] = np.random.randn(H,1) / np.sqrt(H)
  
def preprocess_pong(I):
  # There will always be a preprocessing step, but it will look different for different games.
  # In this case turn 210x160x3 uint8 frame into a 6400 (80x80) 1D float vector.
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  I = I.astype(np.float).ravel()
  I.shape = (6400, 1)
  return I


def policy_forward(x):
  # Neural network begins here. Using multiple hidden layers to show it can be done.
  h1 = relu_hidden_layer(model['W1'], x)
  h2 = relu_hidden_layer(model['W2'], h1)
  # Output layer. This is going to look largely the same if we are wanting two probabilities as our output.
  logitp = np.dot(model['W3'].T, h2)
  p = sigmoid(logitp)
  # Return the number we are interested in (in this case the probability of taking action 2) and the output
  # the hidden states. The latter is not strictly necessary, but it will make our lives easier
  return p, h1.ravel(), h2.ravel()

def policy_backward(xs, h1s, h2s, prob_action_2s, actions, rewards):
  # Implement Andrej's vaguely strange gradient
  actions = (actions - 1) % 2  # 1 if action is 2, 0 otherwise 
  actions = actions.astype('float64') - prob_action_2s

  discounted_rewards = discount_rewards(rewards)
  # Standardize the rewards to be unit normal because Andrej says so.
  discounted_rewards -= np.mean(discounted_rewards)
  discounted_rewards /= np.std(discounted_rewards)
  delta3 = actions * discounted_rewards
  # Right, now we have this strange quantity.
  dW3 = (h2s.T @ delta3).ravel()
  dW3.shape = (dW3.shape[0],1)
  # Do the next layer.
  delta2 = backprop_relu_hidden_layer(delta3, model['W3'], h2s.T)
  dW2 = np.dot(h1s.T, delta2.T)
  delta1 = backprop_relu_hidden_layer(delta2.T, model['W2'], h1s.T)
  dW1 = delta1 @ xs
  return {'W1':dW1, 'W2':dW2, 'W3':dW3}

# in this case we are using rmsprop to update our parameters. this is very model-specific.
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

# again, this function is specific to rmsprop.
def rms_parameter_update(model, grad_buffer, rmsprop_cache):
  for k,v in model.items():
    g = grad_buffer[k] # gradient
    rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
    model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
    grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

if __name__ == '__main__':
    env = gym.make("Pong-v0")
    observation = env.reset()
    env.seed(42)
    env.action_space.seed(24)
    prev_x = None
    xs,h1s,h2s,prob_action_2s,actions,rewards = [],[],[],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 0
    while True:
      if render: env.render()
    
      cur_x = preprocess_pong(observation)
      x = cur_x - prev_x if prev_x is not None else np.zeros(D)
      prev_x = cur_x
    
      # This neural network outputs the probability of taking action 2. This is unusual, normally it would output
      # the probabilities for taking each action. But, if we have two actions, it's not wrong.
      prob_action_2, h1, h2 = policy_forward(x)
      action = 2 if np.random.uniform() < prob_action_2 else 3
    
      xs.append(x.ravel())
      h1s.append(h1)    # We don't strictly need to remember this, but it will make our lives easier
      h2s.append(h2)    # Same
      prob_action_2s.append(prob_action_2)    # Same
      actions.append(action)

      # step the environment and get new measurements
      observation, reward, done, info = env.step(action)
      reward_sum += reward
    
      rewards.append(reward)    # Recall that we must see the outcome of the action before we write down the reward for taking it
    
      if done: # an episode finished
        episode_number += 1
    
        # stack together all inputs, hidden states, probabilities, actions and rewards for this episode
        xs = np.vstack(xs)
        h1s = np.vstack(h1s)    # Both h and prob_action_2 are strictly functions of x, so we don't
        h2s = np.vstack(h2s)    # need to remember them. But we should, because that will be less work.
        prob_action_2s = np.vstack(prob_action_2s) 
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
    
        # Give backprop everything it could conceivably need
        grad = policy_backward(xs, h1s, h2s, prob_action_2s, actions, rewards)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch
        xs,h1s,h2s,prob_action_2s,actions,rewards = [],[],[],[],[],[]
    
        if episode_number % batch_size == 0:
          rms_parameter_update(model, grad_buffer, rmsprop_cache)
    
        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        # I added the if render statement. If we are training the model, we don't want to waste time printing things out
        if render: 
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        # Save the data every third observation.
        if episode_number % 3 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None
      #I added this if render statement for the same reason
      if render:
        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
          print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
else:
    print(__name__)


