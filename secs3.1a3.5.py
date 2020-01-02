#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 19:33:37 2019

@author: sebas
"""


import pymc3 as pm
# import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


model = pm.Model()

with model:
    tita1 = pm.Beta('tita1', 1, 1) #equivalente a uniforme 0,1
    obs1 = pm.Binomial('examen1', 10, tita1, observed=[7,3])
    
    # tita2 = pm.Beta('tita2', 1, 1) #equivalente a uniforme 0,1
    # obs2 = pm.Binomial('examen2', 20, tita1, observed=[16])
    
    # dif = pm.Deterministic('dif', tita1-tita2)

    trace = pm.sample(4000, tune=2000, cores=4)



    
#plt.plot(trace.get_values('tita'))
#plt.show()
pm.traceplot(trace, legend=True)
# pm.rhat(trace)
# pm.forestplot(trace)
# pm.plot_posterior(trace, credible_interval=0.95)
pm.plot_posterior(trace, point_estimate='mean', credible_interval=0.95)
# pm.plot_posterior(trace, var_names=['dif'], ref_val=-0.2, 
                  # kind='hist', round_to=2, credible_interval=0.95)

    
    
# with model:
#     post_pred = pm.sample_posterior_predictive(trace, samples=20000)

# fig, ax = plt.subplots()
# sns.distplot(post_pred['examen'].mean(axis=1), label='Posterior predictive means')#0, ax=ax)
# # ax.axvline(data.mean(), ls='--', color='r', label='True mean')
# ax.legend()