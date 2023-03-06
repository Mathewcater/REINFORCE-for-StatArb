import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch as T
from main_training import *
from Simulator import *

np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino"
})

mblue = (0.098,0.18,0.357)
mred = (0.902,0.4157,0.0196)
mgreen = (0.,0.455,0.247)
mpurple = (0.5804,0.2157,0.9412)
mgray = (0.5012,0.5012,0.5012)
myellow = (0.8,0.8,0)
mwhite = (1.,1.,1.)
cmap = LinearSegmentedColormap.from_list('beamer_cmap', [mred, mwhite, mblue])
colors = [mblue, mred, mgreen, myellow, mpurple, mgray]

# Hyperparams 

num_epochs = 20000
num_states = 20
num_steps = 5
num_eps = 750
n = 10
gamma = 0.7
acts = T.arange(-n,n+1)
states = T.cartesian_prod(T.arange(num_steps + 1), T.arange(num_states+2), acts).float()    
epochs = T.arange(num_epochs)

# Params

vols = [0.3, 0.5, 0.7]
kappas = [4.5, 7.5, 9.5]
        
# === Vary kappa === 

kappa_sims = [Sim(kappa = kappa, sigma = vols[0]) for kappa in kappas]

# Create kappa_bins 

kappa_bins = []
for sim in kappa_sims:
    bins = (sim.mu - ((2.5)*sim.sigma)/(np.sqrt(2*sim.kappa))) + (((2.5)*2*sim.sigma)/(num_states*np.sqrt(2*sim.kappa))) * np.arange(num_states+1)
    bins_ = []
    for i in range(len(bins)):
        bins_.append(format(bins[i], '.2f'))
    bins_.append(format(sim.mu + ((2.5)*sim.sigma)/(np.sqrt(2*sim.kappa)) + ((2.5)*2*sim.sigma)/(num_states*np.sqrt(2*sim.kappa)), '.2f'))
    bins_.reverse()
    kappa_bins.append(bins_)
    
for i in range(len(kappa_bins)):
    for j in range(num_states + 2):
        if j != (num_states + 2)//2 and j!= 0 and j!= num_states + 1:
            kappa_bins[i][j] = ''
            
kappa_avgs = []
kappa_pols = []
kappa_PnLs = []


# === Vary sigma ===

vol_sims = [Sim(kappa = kappas[0], sigma = vol) for vol in vols]

# Create vol_bins

vol_bins = []
for sim in vol_sims:
    bins = (sim.mu - ((2.5)*sim.sigma)/(np.sqrt(2*sim.kappa))) + (((2.5)*2*sim.sigma)/(num_states*np.sqrt(2*sim.kappa))) * np.arange(num_states+1)
    bins_ = []
    for i in range(len(bins)):
        bins_.append(format(bins[i], '.2f'))
    bins_.append(format(sim.mu + ((2.5)*sim.sigma)/(np.sqrt(2*sim.kappa)) + ((2.5)*2*sim.sigma)/(num_states*np.sqrt(2*sim.kappa)), '.2f'))
    bins_.reverse()
    vol_bins.append(bins_)

for i in range(len(vol_bins)):
    for j in range(num_states + 2):
        if j != (num_states + 2)//2 and j!= 0 and j!= num_states + 1:
            vol_bins[i][j] = ''
            
vol_avgs = []
vol_pols = []
vol_PnLs = []
################################################