#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:49:39 2020

@author: Jyanqa
"""
import plotly.graph_objects as go
import Setting as s
#import plotly.express as px
import pandas as pd
#from plotly import io
from statsmodels.base.model import GenericLikelihoodModel
from scipy import stats
import numpy as np
#import statsmodels.api as sm
import scipy.stats as st


class LR(GenericLikelihoodModel):
    def loglike(self, params):
        scale = params[-1]
        weights = params[:-1]
        y_hat = np.sum(weights * self.exog[:, :-1], axis=-1)
        return np.sum(stats.norm.logpdf(self.endog, loc=y_hat, scale=scale))
  
def boot(y,x):
    model = LR(y, x)
    res = model.fit()
    resboot = res.bootstrap(s.noboot) #boot 10 times
    mean = resboot[0]
    std = resboot[1]
    z_score = st.norm.ppf(1 - (1 - s.confi)/2)
    upCI = mean + (z_score*(std/s.noboot))
    lowCI = mean - (z_score*(std/s.noboot))
    bootres = pd.DataFrame({'Average Coefs':mean, 'Upper CI Coefs': upCI, 'Lower CI Coefs':lowCI})
    bounds = (upCI, lowCI)
    return bootres # upCI, lowCI

    









