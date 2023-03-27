from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

# fix chi, since we are using the same value everywhere
chi = 0.4

# set up a dictionary for the equilibrium volume fractions
phip={}

# exxpression for the chemical potential, normalized by kT
def mus(phip, Nel):
    return np.log(1-phip)+phip+chi*phip**2+phip**(1/3)/Nel

# set up axes we'll use to make the plots
fig, ax = plt.subplots(1,1, figsize=(5,3), constrained_layout=True)
ax.set_xlabel(r'$\phi _p$')
ax.set_ylabel(r'$\mu _s /k_B T$')

# set the polymer volume fractions we'll use for the chemical potential plots
p = np.linspace(0, 0.7, 1000)

# now solve for condition where chemical potential is zero and make the plots
for Nel in [3.6, 6]:
    def f_to_solve(phip):
        return mus(phip, Nel)
    
    # obtain the solution, print the result and generate the plots
    phip[Nel]  = fsolve(f_to_solve, 0.8)[0]       
    print(Nel, ', ', phip[Nel], ',', 1/phip[Nel])
    ax.plot(p, f_to_solve(p), '-', label = r'$N_{el}=$'+str(Nel))

# add the legend to the axis and save the figure
ax.legend()
fig.savefig('../figures/swelling_solve.svg')