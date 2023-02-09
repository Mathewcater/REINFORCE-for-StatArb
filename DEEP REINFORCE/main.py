import numpy as np
import matplotlib.pyplot as plt
from DPG import *
from envs import *

np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino"
})

if __name__ == '__main__':
    
    sim = Sim(kappa = 4.5, sigma = 0.3) 
    
    # Average Return Per Epoch Plot
    
    epochs = 1000
    num_states = 20
    num_actions = 10
    bins = list(map(str, np.insert((sim.mu - ((2.5)*sim.sigma)/(np.sqrt(2*sim.kappa))) + (((2.5)*2*sim.sigma)/(num_states*np.sqrt(2*sim.kappa))) * np.arange(num_states+1),0,0.725)))
    for i in range(22):
        if i % 2 != 0:
            bins[i] = ''
            
    avg_ret_deep, t_policy_deep = DEEP_REINFORCE_LIQ(sim, num_eps=300, num_epochs=epochs, M=num_states, n = num_actions)
    
    x = np.arange(epochs)
    plt.plot(x, avg_ret_deep)
    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.title('Average Return Per Epoch from Learned ANN Policy')
    plt.savefig('./train_curve.png')
    plt.show()
    
    # Policy (Heat-Map) Plot
    
    plt.imshow(t_policy_deep.data)
    plt.xticks(np.arange(2*num_actions + 1), np.arange(-num_actions,num_actions + 1))
    plt.yticks(np.arange(num_states + 2), bins)
    plt.xlabel("Inventory")
    plt.ylabel("Price")
    plt.title(r'Learned ANN Policy; $\pi_{\theta}(a\ |\ s)$')
    plt.colorbar()
    plt.savefig('./policy.png')
    plt.show()
    