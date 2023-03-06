import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch as T
from main_training import *
from utils import *
from scipy import stats


np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino"
})

if __name__ == '__main__':
 
    (fig1, ax1), (fig2, ax2), (fig3, axs) = plt.subplots(1, 1, sharey=True), plt.subplots(1, 1, sharey=True), plt.subplots(1, 6, sharey=True)    
    sim = Sim(kappa=4.5, sigma=0.3)
    avgs, pol, PnL = DEEP_REINFORCE(sim, num_eps=num_eps, num_steps=num_steps, num_epochs=num_epochs, n=n)    

    # === Average Return on Epoch Plot ===
    
    ax1.plot(epochs, avgs)
    ax1.set(xlabel='Epochs', ylabel='Average Change in Portfolio Value', title='Average Change in Portfolio Value; Learned ANN Policy')    
    
    
    # === PnL Plots === 
    num_bins = 150
    dom = np.linspace(-7, 40, 1500)
    kde = stats.gaussian_kde(PnL)
    ax2.hist(PnL, density=True, bins=num_bins, color = colors[0], alpha=0.5)
    ax2.plot(dom, kde(dom))
    ax2.set(xlabel=r'P\&L')

    fig2.text(0.04, 0.5, 'Density', va='center', rotation='vertical')
    fig2.suptitle(r'Distribution of Terminal P\&L; Learned ANN Policy')

    # === Policy (Heat-Map) Plot ===
    
    periods = [0, 1, 2, 3, 4, 5]
    
    # === Create x and y-axis labels === 
    bins = (sim.mu - ((2.5)*sim.sigma)/(np.sqrt(2*sim.kappa))) + (((2.5)*2*sim.sigma)/(num_states*np.sqrt(2*sim.kappa))) * np.arange(num_states+1)
    prices = []
    for i in range(len(bins)):
        prices.append(format(bins[i], '.2f'))
    prices.append(format(sim.mu + ((2.5)*sim.sigma)/(np.sqrt(2*sim.kappa)) + ((2.5)*2*sim.sigma)/(num_states*np.sqrt(2*sim.kappa)), '.2f'))
    prices.reverse()
    
    for j in range(num_states + 2):
        if j != (num_states + 2)//2 and j!= 0 and j!= num_states + 1:
            prices[j] = ''
    
    invs = [str(i) for i in range(-n,n+1)]
    for i in range(len(invs)):
        if i % 2 == 1:
            invs[i] = ''
    #############################
    
    # Plot optimal policies through time
    
    for k in range(len(periods)):
        
        learned_pol = T.zeros((num_states+2, len(acts)))
        A = pol[states[:, 0] == periods[k]]
        opt_acts = acts[T.argmax(A, dim=1)]
        
        for i in range(num_states+2):
            for j in range(len(acts)):
                learned_pol[i][j] = opt_acts[i*(len(acts)) + j]

        learned_pol = T.flip(learned_pol, dims=[0])
        axs[k].set_xlabel('Inventory')
        axs[k].set_title(f'Learned; Period: {periods[k]}')
        if k == 0:
            axs[k].set_yticks(T.arange(num_states + 2), labels=prices)
            axs[k].set_ylabel('Price')
            
        divider = make_axes_locatable(axs[k])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(axs[k].imshow(learned_pol), cax=cax)
    
    plt.setp(axs, xticks=np.arange(len(acts)), xticklabels=invs) 
    fig3.suptitle('Temporal Evolution of Learned ANN Policies')
    plt.show()
    
    ###############################