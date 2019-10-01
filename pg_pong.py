""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
    Based on http://karpathy.github.io/2016/05/31/rl/."""
import numpy as np
import pickle
import gym    # For full generality, we might not depend on OpenAI Gym. This is comfortably on the back burner.

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = True

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x))

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def relu_hidden_layer(x, weights):
  retval = weights @ x
  retval[retval<0] = 0
  return retval

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)    # The sigmoid of logp is not p. It just ain't so. Instead the logp variable should be named logitp.
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(hidden_layer_outputs, so_called_gradients):
  """ backward pass. (hidden_layer_outputs is array of intermediate hidden states) """
  # If you had a more complicated neural network, you probably would want to call it h instead of hidden_layer_outputs.
  # But if you had a more complicated neural network, wouldn't you use TensorFlow? It would be much work and much correcting
  # of typos to do it by hand...
  dW2 = np.dot(hidden_layer_outputs.T, so_called_gradients).ravel()    # In the base case with one example, this is just
                                                                       # multiplying the hidden layer outputs by a scalar
  dhidden_layer = np.outer(so_called_gradients, model['W2'])
  dhidden_layer[hidden_layer_outputs <= 0] = 0 # backprop relu
  dW1 = np.dot(dhidden_layer.T, inputs)         # it's not clear how this works given that inputs (the original name was
  return {'W1':dW1, 'W2':dW2}                   # eps) does not appear to be defined in this scope. Trudge on.

if __name__ == '__main__':
    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None # used in computing the difference frame
    xs,hs,so_called_gradient,drs = [],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 0
    while True:
      if render: env.render()
    
      # preprocess the observation, set input to network to be difference image
      cur_x = prepro(observation)
      x = cur_x - prev_x if prev_x is not None else np.zeros(D)
      prev_x = cur_x
    
      # forward the policy network and sample an action from the returned probability
      aprob, h = policy_forward(x)
      action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
    
      # record various intermediates (needed later for backprop)
      xs.append(x) # observation
      hs.append(h) # hidden state
      y = 1 if action == 2 else 0 # a "fake label"
      so_called_gradient.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
      # It isn't really a gradient because it does not satisfy the definition given at https://en.wikipedia.org/wiki/Gradient.
      # Moreover you need to take the gradient with respect to something, and that doesn't happen for a while yet.
      # But once we do that, we will be treating it like a gradient. Hence "so_called_gradient".
    
      # step the environment and get new measurements
      observation, reward, done, info = env.step(action)
      reward_sum += reward
    
      drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    
      if done: # an episode finished
        episode_number += 1
    
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        inputs = np.vstack(xs)
        hidden_layer_outputs = np.vstack(hs)                   # Note that as h is strictly a function of x, we don't actually 
        so_called_gradients = np.vstack(so_called_gradient)    # need to save these. But it will sure be less work if we do.
        epr = np.vstack(drs)
        xs,hs,so_called_gradient,drs = [],[],[],[] # reset array memory
    
        # compute the discounted reward backwards through time
        discounted_rewards = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
    
        so_called_gradients *= discounted_rewards # modulate the gradient with advantage (PG magic happens right here.)
                                                  # Magic is what happens in Harry Potter. 
        # The particular "gradient" chosen means that the changes in the weights are bigger if we pick the action that 
        # was less likely, and also bigger if hidden layer outputs are bigger.
    
        grad = policy_backward(hidden_layer_outputs, so_called_gradients)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch
    
        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
          for k,v in model.items():
            g = grad_buffer[k] # gradient
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
            model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
    
        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        # I added the if render statement. If we are training the model, we don't want to waste time printing things out
        if render: 
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        # Save the data every hundred observations. Why every hundred? Perhaps Andrej liked 100.
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None
      #I added this if render statement for the same reason
      if render:
        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
          print(('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))
else:
    print(__name__)





