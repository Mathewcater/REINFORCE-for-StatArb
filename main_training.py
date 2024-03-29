"""
Created on Fri Mar 10, 10:19:59 2023

@author: matcat
"""
# === IMPORTS === 

import numpy as np
from Simulator import Sim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as T
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import AdamW
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import StepLR
from utils import *

# For reproducibility 
T.manual_seed(543)

# Anamoly Detection 
T.autograd.set_detect_anomaly(True)

# ============ POLICY NETWORK ================== #

class PolicyANN(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(PolicyANN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_SiLU_stack = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, 16),
            nn.SiLU(),
            nn.Linear(16, output_dim)
        )
    
    def forward(self, x):
        return nn.Softmax(dim=-1)(self.linear_SiLU_stack(x))

def DEEP_REINFORCE(sim: Sim, alpha=0.001, num_eps=500, num_steps=50, num_epochs=10000, num_states=20, n=10, phi=0.005, psi=0.5): 
    """Executes training with mini-batch (as opposed to SGD) and employs a fully-connected multi-layered 
    feed forward neural network as policy architecture. Also enforces liquidation at final time step.

    Args:
        sim (Simulator): Simulator object governing price data generating process
        alpha (float): Learning rate
        num_eps (int): Number of episodes per mini-batch
        num_steps (int): Number of (time) steps in a given episode
        num_epochs (int): Number of epochs considered
        n (int): Prescribes num of admissible positions: A = {0, +- 1, ... , +- n}
        phi (float): Transaction fee
        psi (float): Terminal penalty
        num_states (int): Number of prices considered when reporting learned policy 
    Returns:
        avg_rew_epoch (T.tensor): Array consisting of average reward over epsiodes in
                                  each epoch's corresponding mini-batch.
        learned_Policy (T.tensor): The learned policy. A tensor of the distributions over the action
                                   space prescribed by the policy for a collection of prices in
                                   [mu - (2.5)*sigma/sqrt(2*kappa), mu + (2.5)*sigma/sqrt(2*kappa)]
            
        PnL (T.tensor): A collection of terminal PnL's produced by trading with the learned policy
                        on 1000 sample paths.
    """
    pi = PolicyANN(input_dim=3, output_dim=2*n+1)
    optimizer = AdamW(params=pi.parameters(), lr=alpha, maximize=True)
    scheduler = T.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.99)
    acts = T.arange(-n,n+1)  # The action space: {0, +- 1, ... , +- n}                                                           
    avg_rew_epoch = T.zeros(num_epochs) # To be populated with average return on each epoch

    for m in tqdm(range(num_epochs + 1)):
        
        # ===== Generate Mini-Batch of Episodes =====         
        if m == num_epochs: # After training, estimate PnL distribution with 30,000 full episodes. With 
                            # (deterministic) initial price sim.mu and initial inventory 0.
            num_eps = 30_000
            times = T.arange(num_steps + 1).repeat(num_eps,1)/num_steps # normalize time data
            act_sims, act_log_probs = T.zeros((num_eps, num_steps + 2)), T.zeros((num_eps, num_steps))
            prices = sim.paths(num_eps, num_steps, 1/num_steps, deep=True, det=False)
            norm_prices = (prices - sim.mu) / 0.5 # normalize price data
            state_sims = T.stack([times, norm_prices, act_sims[:,1:]])
            
        else:    
            times = T.arange(num_steps + 1).repeat(num_eps,1)/num_steps # normalize time data
            act_sims, act_log_probs = T.zeros((num_eps, num_steps + 2)), T.zeros((num_eps, num_steps))
            prices = sim.paths(num_eps, num_steps, 1/num_steps, deep=True, det=False)
            norm_prices = (prices - sim.mu) / 0.5 # normalize price data
            
            # Randomise initial inventory (uniformly)
            probs = (1/(len(acts)))*T.ones(len(acts)).repeat(num_eps, 1).float()
            dist = Categorical(probs)
            idx = dist.sample()
            state_sims = T.stack([times, norm_prices, act_sims[:,1:]])
            state_sims[2,:,0] = acts[idx]/T.max(acts)
            act_sims[:,0] = acts[idx]
            
        for j in range(num_steps): 
            probs = pi(state_sims[:,:,j].clone().T)
            dist = Categorical(probs)
            idx = dist.sample()
            act_sims[:,j+1] = acts[idx]
            act_log_probs[:,j] = dist.log_prob(idx)
            state_sims[2,:,j+1] = acts[idx]/T.max(acts)
            
        # == Generate Reward Simulations ==
        
        rew_sims = act_sims[:,1:-1]*(prices[:,1:] - prices[:,:-1]) - (phi*(((act_sims[:,:-1])[:,1:] - (act_sims[:,:-1])[:,:-1]) ** 2)) 
        rew_sims[:,-1] -= psi*((act_sims[:,-2]) ** 2) # impose terminal penalty
        G = T.flip(T.cumsum(T.flip(rew_sims, dims=[1]), dim=1), dims=[1]) # matrix of rewards to go (returns)
        PnL = G[:, 0] # Cumulative rewards

        if m != num_epochs:
            avg_rew_epoch[m] = T.sum(rew_sims)/num_eps  # Store average return over mini-batch
            optimizer.zero_grad()
            loss = T.sum((G * act_log_probs))/num_eps   # Compute Loss and Make Backward Pass 
            loss.backward()
            optimizer.step()
            scheduler.step()

    prices = (((sim.mu - ((2.5)*(sim.sigma)/(np.sqrt(2*sim.kappa)))) + ((2.5)*(2*sim.sigma)/(num_states*np.sqrt(2*sim.kappa)))*T.arange(num_states+2)) - sim.mu) / 0.5 
    states = T.cartesian_prod(T.arange(num_steps + 1)/num_steps, prices, acts/T.max(acts)).float()  
    learned_pol = pi(states) # Store Learned Policy 
    learned_pol[states[:, 0] == 1.0] = (F.one_hot(T.tensor(n), num_classes = len(acts)).float()).repeat(len(learned_pol[states[:, 0] == 1.0]), 1) # Enforce liquidation
      
    return avg_rew_epoch, learned_pol, PnL