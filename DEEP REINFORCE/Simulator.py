import numpy as np
import torch
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino"
})
from matplotlib.lines import Line2D
from scipy.stats import norm

class Sim():
    """OU process sample path simulator. Simulates: dX_t = (kappa)(mu - X_t)dt + (sigma)dW_t
    """
    def __init__(self, x0 = 1, mu = 1, kappa = 3, sigma = np.sqrt(6)/10):
    
        self.x0 = x0
        self.kappa = kappa
        self.sigma = sigma
        self.mu = mu

    def paths(self, npaths: int, nsteps: float, dt: float, deep=False, det = True):
    
        """Generates an (npaths) x (nsteps) array with each row being a given path
        and each column being the path at a given time step.

        Returns:
            np.array: array of sample paths
        """
        if deep: 
            p = self.x0 * torch.ones((npaths, nsteps+1))
            if det == False:
                p[:, 0] = torch.normal(mean=self.mu * torch.ones(npaths), std=(self.sigma**2)/2*self.kappa * torch.ones(npaths))
            for i in range(1, nsteps+1):
                p[:, i] = p[:, i-1] + (self.kappa)*(self.mu - p[:, i-1])*dt + self.sigma*np.sqrt(dt)*torch.randn(npaths)
            return p 
        else:
            p = self.x0*np.ones((npaths, nsteps+1))
            if det == False:
                p[:, 0] = np.random.normal(self.mu, (self.sigma**2)/2*self.kappa, npaths)
            for i in range(1, nsteps+1):
                p[:, i] = p[:, i-1] + (self.kappa)*(self.mu - p[:, i-1])*dt + self.sigma*np.sqrt(dt)*np.random.normal(0, 1, npaths)
            return p 

        
if __name__ == "__main__":
    
    md = Sim(kappa = 4.5, sigma = 0.3)
    num_steps = 1000
    num_paths = 1000
    d = md.paths(num_paths, num_steps, 1/num_steps, det = False, deep = True)
    for i in tqdm(range(num_paths)): 
        plt.plot((1/num_steps)*np.arange(num_steps+1), d[i])
    plt.show()
    