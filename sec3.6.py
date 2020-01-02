#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 21:08:22 2019

@author: sebas
"""

import pymc3 as pm
# import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


m = 5
nmax = 500

yobs = [ 16, 18, 22, 25, 28]
# yobs = [20,21,22]*10
# maxy = max(yobs)
# maxy = 35

model = pm.Model()

with model:
    
    tita = pm.Beta('tita', 1, 1) #equivalente a uniforme 0,1
    # cant = pm.Categorical('cantidad', [0]*maxy+[1/(nmax)]*(nmax))
    cant = pm.DiscreteUniform('cantidad', lower=1, upper=nmax)
    # cant = pm.Uniform('cantidad', lower=1, upper=nmax)
    
    obs = pm.Binomial('encuestas', cant, tita, observed=yobs)
    
    trace = pm.sample(10000, tune=101000, cores=4)
    # trace = pm.sample()
    # trace = pm.sample(1000, tune=1000, cores=4, 
                      # nuts_kwargs = {'target_accept' : 0.99})



    
#plt.plot(trace.get_values('tita'))
#plt.show()
pm.traceplot(trace, legend=True)
# pm.rhat(trace)
# pm.forestplot(trace)
# pm.plot_posterior(trace, credible_interval=0.95, compact=True)
# pm.plot_posterior(trace, point_estimate='median', kind='hist')
# pm.plot_posterior(trace, var_names=['dif'], ref_val=-0.2, 

pm.plot_joint(trace)
# pm.summary(trace)
# pm.model_graph.model_to_graphviz(model)
# pm.plot_pair(trace)

# pm.autocorrplot(trace)