import numpy as np
import torch
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

class Sim():
    """OU process sample path simulator. Simulates: dX_t = (kappa)(mu - X_t)dt + (sigma)dW_t
    """
    def __init__(self, x0=1, mu=1, kappa=2, sigma=np.sqrt(2)/10):
    
        self.x0 = x0
        self.kappa = kappa
        self.sigma = sigma
        self.mu = mu

    def paths(self, npaths: int, nsteps: int, dt: float, deep=False, det = True):
    
        """Generates an (npaths) x (nsteps) array with each row being a given path
        and each column being the path at a given time step.

        Returns:
            np.array or torch.tensor: array/tensor of sample paths
        """
        if deep: 
            p = self.x0 * torch.ones((npaths, nsteps+1))
            if det == False:
                p[:, 0] = torch.normal(mean=self.mu * torch.ones(npaths), std=((self.sigma)/((2*self.kappa)**(0.5))) * torch.ones(npaths))
            for i in range(1, nsteps+1):
                p[:, i] = p[:, i-1] + (self.kappa)*(self.mu - p[:, i-1])*dt + self.sigma*np.sqrt(dt)*torch.randn(npaths)
            return p 
        else:
            p = self.x0*np.ones((npaths, nsteps+1))
            if det == False:
                p[:, 0] = np.random.normal(self.mu, (self.sigma)/((2*self.kappa**(0.5))), npaths)
            for i in range(1, nsteps+1):
                p[:, i] = p[:, i-1] + (self.kappa)*(self.mu - p[:, i-1])*dt + (self.sigma*np.sqrt(dt))*np.random.normal(0, 1, npaths)
            return p 

        
if __name__ == "__main__":
    
    # Produces plots of price paths, each using Euler scheme 
    # simulating the process out to 1 second.
    
    sim = Sim(kappa=4.0, sigma=0.9)
    num_steps = 5
    num_paths = 5
    
    pths = sim.paths(num_paths, num_steps, 1/num_steps, det=True, deep=False)
    for i in tqdm(range(num_paths)): 
        alpha = 0.05
        if i == num_paths-1:
            alpha = 1
        plt.plot((1/num_steps)*np.arange(num_steps+1), pths[i], alpha=alpha)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Sample Paths of Price')
    plt.show()