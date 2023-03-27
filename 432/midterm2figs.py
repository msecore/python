#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 20:56:47 2022

@author: ken
"""
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

fig, ax = plt.subplots(1,1, figsize=(5,3), constrained_layout=True)

ax.set_xlim([1,5])
ax.set_xlabel(r'$\lambda$')

ax.set_ylim([0, 60])
ax.set_ylabel(r'$\sigma_{eng}$ (MPa)')

fig.savefig('../figures/eng_stress_strain.svg')

ax.set_ylabel(r'$\sigma_{true}$ (MPa)')

fig.savefig('../figures/true_stress_strain_blank.svg')

ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'survival probability')
ax.set_ylim([0,1])
ax.set_xticks([])

fig.savefig('../figures/blank_weibull_plot.svg')

#%%
plt.close('all')
fig, ax = plt.subplots(1,2, figsize=(8,3), constrained_layout=True)
stress_norm = 2*1.38e-23*300/500e-30

stress = np.linspace(0, 3*stress_norm, 100)
strain_rate = 1e-7*np.sinh(stress/stress_norm)
ax[0].plot(stress, strain_rate, '-')
ax[1].semilogy(stress, strain_rate, '-')
for k in [0,1]:
    ax[k].set_xlabel(r'$\sigma$ (Pa)')
    ax[k].set_ylabel(r'strain rate (s$^{-1}$)')

ax[0].set_ylim(bottom=0)
    
fig.savefig('../figures/creep_data.svg')
