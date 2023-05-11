import numpy as np 
import torch
from mab import eps_bandit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Deep Q-network.
"""
class DQN(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super(DQN, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_actions))
        return 

    def forward(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, device=DEVICE)
        y = self.net(x.float())
        return y

class contextual_bandit(eps_bandit):
    '''
    contextual epsilon-greedy k-bandit problem
    
    Inputs
    =====================================================
    k: number of arms (int)
    eps: probability of random action 0 < eps < 1 (float)
    iters: number of steps (int)
    mu: set the average rewards for each of the k-arms.
        Set to "random" for the rewards to be selected from
        a normal distribution with mean = 0. 
        Set to "sequence" for the means to be ordered from 
        0 to k-1.
        Pass a list or array of length = k for user-defined
        values.
    '''
    
    def __init__(self, k, eps, iters, hidden_size, state_size, mu='random', learning_rate=0.1, qnet=None):
        # Number of arms
        self.k = k
        # Search probability
        self.eps = eps
        # Number of iterations
        self.iters = iters
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(k)
        
        # Total mean reward
        self.mean_reward = 0
        
        # Mean reward for each arm
        self.reward = np.zeros(iters)
        
        self.hidden_size=hidden_size
        self.state_size=state_size
        
        self.neuralNet = DQN(state_dim=self.state_size, hidden_dim=self.hidden_size, num_actions=self.k).to(DEVICE)
        self.opt = torch.optim.Adam(params=self.neuralNet.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, k)
        elif mu == 'sequence':
            # Increase the mean for each arm by one
            self.mu = np.linspace(0, k-1, k)

        np.random.seed(13)
        torch.manual_seed(13)
        
    def act(self, state, eps = 0):
        with torch.no_grad():
            self.neuralNet.eval()
            q_values = self.neuralNet(state)
        # Random action.
        if np.random.rand() < eps:
            nA = q_values.shape[-1]
            action = np.random.choice(nA)
        # Greedy policy.
        else:
            print('EXPLOIT')
            action = q_values.argmax().cpu().numpy()
        return action
    
    def learn(self, reward, state, action):
        # Update counts
        self.n += 1
        self.k_n[action] += 1
        
        # r(s) = reward 
        # Q(a,s) = q_values[action] 
        # max... = y 



        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        #print(self.mean_reward)

        self.neuralNet.train()
        q_values = self.neuralNet(state)
        y = q_values.max()

        # Update results for a_k
        with torch.no_grad():
            # yhat = y + (reward - y) / self.k_n[action]
            # print(type(action))
            # print(type(q_values.detach().numpy()))

            # print(type(y))
            yhataction = float(action)

            yhat = q_values[action] + yhataction *(reward + 0.9*(y))-q_values[action]
            # yhat = yhat.reshape(y.shape).float()
        
        # # Back-propagation.
        self.opt.zero_grad()
        loss = self.loss_fn(y, yhat)
        print(loss)
        loss.backward()
        self.opt.step()
        
    def incrementIters(self, increment): 
        self.iters += int(increment)
        return

    def reset(self):
        # Resets results while keeping settings
        self.n = 0
        self.k_n = np.zeros(self.k)
        self.mean_reward = 0
        self.reward = np.zeros(self.iters)
        self.neuralNet = np.zeros(self.k)
        return