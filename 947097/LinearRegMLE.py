#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:33:19 2020

@author: Jyanqa
"""
import matplotlib.pyplot as plt
import Setting as s
import plotly.express as px
import pandas as pd
from plotly import io
from Import_Data import load_data
from statsmodels.base.model import GenericLikelihoodModel
from scipy import stats
import numpy as np
import statsmodels.api as sm

#io.renderers.default = 'browser'


class LR(GenericLikelihoodModel):
    def loglike(self, params):
        scale = params[-1]
        weights = params[:-1]
        y_hat = np.sum(weights * self.exog[:, :-1], axis=-1)
        return np.sum(stats.norm.logpdf(self.endog, loc=y_hat, scale=scale))
  
def fitLR(y,x):
    model = LR(y, x)
    res = model.fit()
    return res


def fitOLS(y,x):
    ols = sm.OLS(y, x)
    ols_res = ols.fit()
    return ols_res


def l1(y,x):
   dfy, dfx = load_data(s.dataset)
   model = LR(y, x)
   res = model.fit()
   y_predicted =np.dot(x,res.params).reshape(-1, 1)
   plt.hist(y_predicted)
   l1 = np.sum(np.abs(y-y_predicted)) 
   return l1


    
    
    
    
    
    
    
    
    
    
    
    
    






