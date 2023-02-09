# === IMPORTS === 

import numpy as np
from Simulator import Sim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch as T
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import Adam
from torch.distributions.categorical import Categorical


# For reproducibility 

T.manual_seed(42)
T.autograd.set_detect_anomaly(True)


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino"
})

def init_zero(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 0.0)
        nn.init.constant_(m.bias, 0.0)
        
# ============ POLICY NETWORK ================== #

class PolicyANN(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(PolicyANN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_SiLU_stack = nn.Sequential(
            nn.Linear(input_dim, 30),
            nn.SiLU(), 
            nn.Linear(30, 30),
            nn.SiLU(),  
            nn.Linear(30, output_dim),
        )
    
    def forward(self, x):
        return nn.Softmax(dim=-1)(self.linear_SiLU_stack(x))

def DEEP_REINFORCE_LIQ(sim: Sim, alpha = 0.001, num_eps = 100, num_steps = 50, num_epochs = 10000, M = 20, n = 10):
    """Executes training with mini-batch as opposed to SGD and employs a fully-connected multi-layered 
    feed forward neural network as Policy architecture. Also enforces liquidation at final time step.

    Args:
        alpha (float): Learning rate
        num_eps (int): Number of episodes per mini-batch
        num_steps (int): Number of (time) steps in a given episode
        num_epochs (int): Number of epochs considered
        M (int): Number of M in binning of state space
        n (int): Prescribes num of admissible positions: {0, +- 1, ... , +- n}
    Returns:
        avg_rew_epoch (T.tensor): Array consisting of average reward over epsiodes in
                                      each epoch's corresponding mini-batch.
        t_Policy (T.tensor): The trained Policy. Two dimensional tensor with i,j-th entry
                                 equal to the probability that action j is selected in state i
                                 prescribed by the trained/learned policy.                 
    """
    model = PolicyANN(input_dim=3, output_dim=2*n+1)
    optimizer = Adam(params=model.parameters(), lr=alpha, maximize = True)    
    phi = 0.005  # Transaction fee
    bins = (sim.mu - ((2.5)*sim.sigma)/(np.sqrt(2*sim.kappa))) + (((2.5)*2*sim.sigma)/(M*np.sqrt(2*sim.kappa))) * T.arange(M+1) 
    acts = T.arange(-n,n+1)  # The action space: {0, +- 1, ... , +- n}                                                           
    avg_rew_epoch = T.zeros(num_epochs) # To be populated with average return on each epoch
    

    for m in tqdm(range(num_epochs)):
        
        # ===== Generate Mini-Batch of Episodes ===== 
        
        # == Generate Price Simulations == 
        
        pths = sim.paths(num_eps, num_steps, 1/num_steps, deep = True, det = False) # (num_eps, num_steps + 1)
        price_sims = T.bucketize(pths, bins) # (num_eps, num_steps + 1)  
        print(price_sims.size())
        # == Generate Action Simulations ==
        times = T.arange(num_steps + 1).repeat((num_eps,1))
        print(times.size())
        
        act_sims = T.zeros((num_eps, num_steps + 1))
        act_log_probs = T.zeros((num_eps, num_steps + 1))
        state_sims = T.stack([times, price_sims, act_sims])
        
        
        
        # for i in range(num_eps):
        #     for j in range(num_steps+1):
        #         if j == 0:
        #             probs = model(T.tensor([j, price_sims[i][j], 0]).float())
        #         elif j == num_steps: # Enforce Liquidation at Final Period
        #             probs = F.one_hot(T.tensor([n]), num_classes = len(acts)).view(len(acts))
        #         else:
        #             probs = model(T.tensor([j, price_sims[i][j], act_sims[i][j-1]]).float())
        #         dist = Categorical(probs)
        #         idx = dist.sample()
        #         act_sims[i][j] = acts[idx]
        #         act_log_probs[i][j] = dist.log_prob(idx)
        
        
        
        for j in range(num_steps + 1):
    
            if j == num_steps: # Enforce Liquidation 
                probs = F.one_hot(T.tensor([n]), num_classes = len(acts)).view(len(acts)).repeat(num_eps, 1)
            else:
                probs = model(T.transpose(state_sims[:,:,j],0,1))
            
            dist = Categorical(probs)
            idx = dist.sample()
            act_sims[:,j] += acts[idx]
            act_log_probs[:,j] += dist.log_prob(idx) 
            if j != 0:
                state_sims[2,:,j] += act_sims[:,j-1]


        print(act_sims)
        
        # == Generate Reward Simulations ==
        
        rew_sims = act_sims[:,:-1]*(pths[:,1:] - pths[:,:-1]) - phi * T.abs(act_sims[:,1:] - act_sims[:,:-1])
        
        avg_rew_epoch[m] += T.sum(rew_sims)/num_eps  
        
        # == Compute Loss and Make Backward Pass == 
        
        G = T.flip(T.cumsum(T.flip(rew_sims, dims=[1]), dim=1), dims=[1]) # (num_eps, num_steps)
        loss = T.sum((G * act_log_probs[:,:-1]))/num_eps
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                  
    states = T.cartesian_prod(T.arange(num_steps + 1), T.arange(M+2), acts)  # (M+1, 1)
    t_Policy = model(states) # (M+1, output_dim)
        
    return avg_rew_epoch, t_Policy