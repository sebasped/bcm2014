#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 16:57:00 2019

@author: sebas
"""


import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns



model = pm.Model()
# y = pm.Normal.dist(mu=2, sigma=2.5)
# yobs = y.random(size=40)
yobs = [-27.02, 3.57, 8.191, 9.898, 9.603, 9.945, 10.056]
n = len(yobs)


with model:
    
    mu = pm.Normal('mu', mu=0, tau=0.001)
    tau1 = pm.Gamma('tau1', alpha=0.001, beta=0.001, shape=n)
    
    sd1 = pm.Deterministic('sd1', 1/pm.math.sqrt(tau1) ) 
    
    obs = pm.Normal('obs', mu=mu, tau=tau1, observed=yobs)
    
    trace = pm.sample(10000, tune=4000, cores=4)
    # trace = pm.sample()
    # trace = pm.sample(1000, tune=1000, cores=4, 
                      # nuts_kwargs = {'target_accept' : 0.99})



    
#plt.plot(trace.get_values('tita'))
#plt.show()
pm.traceplot(trace, legend=True)
# pm.rhat(trace)
# pm.forestplot(trace)
pm.plot_posterior(trace, var_names=['mu'], credible_interval=0.95)
# pm.plot_posterior(trace, var_names=['sd1', 'sd2', 'sd3'], 
                  # credible_interval=0.95)
# pm.plot_posterior(trace, var_names=['sd4', 'sd5', 'sd6', 'sd7'], 
                  # credible_interval=0.95)
# pm.plot_posterior(trace, var_names=['tau5'], credible_interval=0.95)


# pm.plot_posterior(trace, point_estimate='median', kind='hist')
# pm.plot_posterior(trace, var_names=['dif'], ref_val=-0.2, 

# pm.plot_joint(trace)
# pm.summary(trace)
# pm.model_graph.model_to_graphviz(model)
# pm.plot_pair(trace)

# pm.autocorrplot(trace)