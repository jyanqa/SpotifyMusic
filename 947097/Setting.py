#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:33:18 2020

@author: Jyanqa
"""
from plotly import io
io.renderers.default = 'browser'

PLOT = True

dataset = './data.csv'
y = ['popularity']
x = ['acousticness', 'danceability']
#---
confi = 0.95
noboot = 10

#---

preprocessing = False
n_components = 2
#clustering_methods = ['kmeans'] #, 'gmm'

