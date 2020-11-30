#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:08:05 2020

@author: Jyanqa
"""
#from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from sklearn.metrics import classification_report
#import numpy as np
#from scipy.stats import mode

def kmeans_cluster(df, n_components):
    kmeans = KMeans(n_clusters=n_components)
    kmeans.fit(df)
    return kmeans.predict(df)


def classification_measure(true_values, pred_values):
   # pred_values = collapse_components(true_values, pred_values)

    print(classification_report(true_values, pred_values))