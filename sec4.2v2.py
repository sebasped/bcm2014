#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 17:32:29 2019

@author: sebas
"""


import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns



model = pm.Model()
# y = pm.Normal.dist(mu=2, sigma=2.5)
# yobs = y.random(size=40)
upb = 1000
with model:
    
    mu = pm.Normal('mu', mu=0, tau=0.001)
    sd1 = pm.Uniform('sd1', lower=0, upper=upb)
    sd2 = pm.Uniform('sd2', lower=0, upper=upb)
    sd3 = pm.Uniform('sd3', lower=0, upper=upb)
    sd4 = pm.Uniform('sd4', lower=0, upper=upb)
    sd5 = pm.Uniform('sd5', lower=0, upper=upb)
    sd6 = pm.Uniform('sd6', lower=0, upper=upb)
    sd7 = pm.Uniform('sd7', lower=0, upper=upb)
    
    
    obs1 = pm.Normal('obs1', mu=mu, sigma=sd1, observed=[-27.02])
    obs2 = pm.Normal('obs2', mu=mu, sigma=sd2, observed=[3.57])
    obs3 = pm.Normal('obs3', mu=mu, sigma=sd3, observed=[8.191])
    obs4 = pm.Normal('obs4', mu=mu, sigma=sd4, observed=[9.898])
    obs5 = pm.Normal('obs5', mu=mu, sigma=sd5, observed=[9.603])
    obs6 = pm.Normal('obs6', mu=mu, sigma=sd6, observed=[9.945])
    obs7 = pm.Normal('obs7', mu=mu, sigma=sd7, observed=[10.056])
    
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