#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:27:30 2019

@author: sebas
"""

import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns



model = pm.Model()
y = pm.Normal.dist(mu=2, sigma=2.5)
yobs = y.random(size=40)
with model:
    
    mu = pm.Normal('mu', mu=0, tau=0.001)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    
    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=yobs)
    
    trace = pm.sample(10000, tune=4000, cores=4)
    # trace = pm.sample()
    # trace = pm.sample(1000, tune=1000, cores=4, 
                      # nuts_kwargs = {'target_accept' : 0.99})



    
#plt.plot(trace.get_values('tita'))
#plt.show()
pm.traceplot(trace, legend=True)
# pm.rhat(trace)
# pm.forestplot(trace)
pm.plot_posterior(trace, credible_interval=0.95)
# pm.plot_posterior(trace, point_estimate='median', kind='hist')
# pm.plot_posterior(trace, var_names=['dif'], ref_val=-0.2, 

pm.plot_joint(trace)
# pm.summary(trace)
# pm.model_graph.model_to_graphviz(model)
# pm.plot_pair(trace)

# pm.autocorrplot(trace)