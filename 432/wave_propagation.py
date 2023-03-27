from sympy import symbols, Matrix, exp, diff, KroneckerDelta, sqrt, eye, lambdify
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

I = {} # dictionary of invarients
lam = symbols('lambda', positive=True)
lamx, lamy, lamz = symbols(['lambda_x', 'lambda_y', 'lambda_z'] ,positive = True)

G, Jstar = symbols(['G', 'J^*'], positive=True)
ivals = [1]
for i in ivals:
    I[1] = symbols('I'+str(i), positive = True)

# deformation gradient tensor for stretchin the x direction
F = Matrix([[lamx, 0, 0],
               [0, 1/sqrt(lamx), 0],
               [0, 0, 1/sqrt(lamx)]])

W = (G*Jstar/2)*(exp((I[1]-3)/Jstar)-1) # strain energy function

# Build dictionaries with the first and second derivatives
deriv = {'W_I':{}, 'I_F':{}}
deriv2 = {'W_I':{}, 'I_F':{}}

for i in list(I.keys()):
    deriv['W_I'][i] = diff(W, I[i])
    deriv2['W_I'][i] = deriv['W_I'][i]

# enter derivatives of I by hand
deriv['I_F'][1] = 2*F  # Eq. 3.22

# calculate true stress
stress = F*deriv['W_I'][1]*deriv['I_F'][1] # Eq 2.26

# now include the pressure term, from the fact that normal stresses are zero in
# transverse direction
stress = stress - eye(3)*stress[1,1]

# use lamx as independent varible
stress = stress.subs(I[1], lamx**2+2/lamx)

A = np.empty((3,3,3,3), dtype=object) # undeformed configuraion
A0 = np.empty((3,3,3,3), dtype=object) # deformed configuration
deriv2['I_F'][1] = np.empty((3,3,3,3), dtype=object)

idx = np.arange(3) # used for nested loops

# calculate coeffients in undeformed configuration: Eq. 3.21
for i, a, j, b in product(idx, idx, idx, idx):
    deriv2['I_F'][1][i, a, j, b] = (2*KroneckerDelta(i,j)*
                                           KroneckerDelta(a, b))
    A[i, a, j, b] = (deriv2['W_I'][1]*deriv['I_F'][1][i, a]*deriv['I_F'][1][j,b] +
                            deriv['W_I'][1]*deriv2['I_F'][1][i,a,j,b])
    
# now calculate coefficients of A in deformed configuraiton
nonzero_list=[] # list of nonzero elments of A0
for p, i, q, j in product(idx, idx, idx, idx):
    A0[p,i,q,j] = 0
    for a, b in product(idx, idx):  # Eq. 2.24
        A0[p, i, q, j] += F[p,a]*F[q,b]*A[a, i, b, j]
    if A0[p,i,q,j]!=0:
        nonzero_list.append([p,i,q,j])
    # rewrite I1 in terms of lamx
    A0[p,i,q,j] = A0[p,i,q,j].subs(I[1], lamx**2+2/lamx)
    
# create functions for normalized stress and moduli
G_perp = A0[2,1,2,1]
G_parallel = A0[0,1,0,1]

funcs = {
    'G_perp_norm': lambdify([lamx, Jstar], G_perp/G),
    'G_parallel_norm': lambdify([lamx, Jstar], G_parallel/G),
    'stress':lambdify([lamx, Jstar], stress[0,0]/G)
    }

# now make the plots
plt.close('all')
fig, ax = plt.subplots(1, 3, figsize=(7,3), constrained_layout = True)
lam = np.linspace(1, 2, 100)

#label the axes
for k in [0,1]: ax[k].set_xlabel(r'$\lambda_x$')
ax[2].set_xlabel(r'$\sigma /G$')
ax[0].set_ylabel(r'$\sigma /G$')
for k in [1,2]: ax[k].set_ylabel(r'$G_{\mathrm{inc}}$')

# make plots for different values of jstar
p = 0
for jstarval in [np.inf, 3.5]:
    ydict = {}
    for ydata in ['G_perp_norm', 'G_parallel_norm', 'stress']:
        ydict[ydata] = funcs[ydata](lam, jstarval)
    
    # plot the data
    ax[0].plot(lam, ydict['stress'], 'C'+str(p)+'-',
               label = r'$J^*=$' + str(jstarval))
    ax[1].plot(lam, ydict['G_perp_norm'], 'C'+str(p)+'--',
               label = r'$\perp$'+' '+str(jstarval))
    ax[1].plot(lam, ydict['G_parallel_norm'], 'C'+str(p)+'-',
               label = r'$\parallel$'+' '+str(jstarval))
    ax[2].plot(ydict['stress'], ydict['G_perp_norm'], 'C'+str(p)+'--',
               label = r'$\perp$'+' '+str(jstarval))
    ax[2].plot(ydict['stress'], ydict['G_parallel_norm'], 'C'+str(p)+'-',
               label = r'$\parallel$'+' '+str(jstarval))  
    p = p+1

# add the legends
for k in [0,1,2]: ax[k].legend()


fig.show()
fig.savefig('../figures/strain_hardening_plots.svg')

        





