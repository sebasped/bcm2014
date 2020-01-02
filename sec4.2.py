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

with model:
    
    mu = pm.Normal('mu', mu=0, tau=0.001)
    tau1 = pm.Gamma('tau1', alpha=0.001, beta=0.001)
    tau2 = pm.Gamma('tau2', 0.001, 0.001)
    tau3 = pm.Gamma('tau3', 0.001, 0.001)
    tau4 = pm.Gamma('tau4', 0.001, 0.001)
    tau5 = pm.Gamma('tau5', 0.001, 0.001)
    tau6 = pm.Gamma('tau6', 0.001, 0.001)
    tau7 = pm.Gamma('tau7', 0.001, 0.001)
    
    sd1 = pm.Deterministic('sd1', 1/pm.math.sqrt(tau1) ) 
    sd2 = pm.Deterministic('sd2', 1/pm.math.sqrt(tau2) ) 
    sd3 = pm.Deterministic('sd3', 1/pm.math.sqrt(tau3) ) 
    sd4 = pm.Deterministic('sd4', 1/pm.math.sqrt(tau4) ) 
    sd5 = pm.Deterministic('sd5', 1/pm.math.sqrt(tau5) ) 
    sd6 = pm.Deterministic('sd6', 1/pm.math.sqrt(tau6) ) 
    sd7 = pm.Deterministic('sd7', 1/pm.math.sqrt(tau7) ) 
    
    obs1 = pm.Normal('obs1', mu=mu, tau=tau1, observed=[-27.02])
    obs2 = pm.Normal('obs2', mu=mu, tau=tau2, observed=[3.57])
    obs3 = pm.Normal('obs3', mu=mu, tau=tau3, observed=[8.191])
    obs4 = pm.Normal('obs4', mu=mu, tau=tau4, observed=[9.898])
    obs5 = pm.Normal('obs5', mu=mu, tau=tau5, observed=[9.603])
    obs6 = pm.Normal('obs6', mu=mu, tau=tau6, observed=[9.945])
    obs7 = pm.Normal('obs7', mu=mu, tau=tau7, observed=[10.056])
    
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
pm.plot_posterior(trace, var_names=['sd1', 'sd2', 'sd3'], 
                  credible_interval=0.95)
pm.plot_posterior(trace, var_names=['sd4', 'sd5', 'sd6', 'sd7'], 
                  credible_interval=0.95)
pm.plot_posterior(trace, var_names=['tau5'], credible_interval=0.95)


# pm.plot_posterior(trace, point_estimate='median', kind='hist')
# pm.plot_posterior(trace, var_names=['dif'], ref_val=-0.2, 

# pm.plot_joint(trace)
# pm.summary(trace)
# pm.model_graph.model_to_graphviz(model)
# pm.plot_pair(trace)

# pm.autocorrplot(trace)