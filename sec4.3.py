#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 21:43:52 2020

@author: sebas
"""


import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns



model = pm.Model()
yobs = [[90,95,100], [105,110,115], [150,155,160]]
npeople = len(yobs)
# ntest = len(yobs[0])

with model:
    
    # el shape define una matriz de RVs. En este caso de 3x1
    # pues así está definido el yobs (las observaciones)
    mu = pm.Uniform('mu', lower=0, upper=300, shape=(npeople,1) )
    sigma = pm.Uniform('sigma', 0, 100)
    
    precision = pm.Deterministic('precision', 1/(sigma**2) )
        
    obs = pm.Normal('obs', mu=mu, tau=1/(sigma**2), observed=yobs)
    
    trace = pm.sample(4000, tune=2000, cores=4)
    # trace = pm.sample()
    # trace = pm.sample(1000, tune=1000, cores=4, 
                      # nuts_kwargs = {'target_accept' : 0.99})



    
pm.traceplot(trace, legend=True)
pm.plot_posterior(trace, var_names=['mu'], credible_interval=0.95)
pm.plot_posterior(trace, var_names=['sigma'], credible_interval=0.95)
pm.plot_posterior(trace, var_names=['precision'], credible_interval=0.95)


#ejercicio 4.3.1
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=10000)

fig, ax = plt.subplots()
sns.distplot(post_pred['obs'].mean(axis=2)[:,0], label='Posterior mu1')#0, ax=ax)
sns.distplot(post_pred['obs'].mean(axis=2)[:,1], label='Posterior mu2')
sns.distplot(post_pred['obs'].mean(axis=2)[:,2], label='Posterior mu3')

# # ax.axvline(data.mean(), ls='--', color='r', label='True mean')
# ax.legend()
