#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:35:54 2020

@author: Jyanqa
"""
import Setting as s
from Import_Data import load_data
import LinearRegMLE as lr
import bootstrap as bt
from clusterize import kmeans_cluster, classification_measure
#---------------------------------------------    
def LRMLE():
    #load data
    dfy, dfx = load_data(s.dataset)
    #fit MLE Regresssion model
    lrmodel = lr.fitLR(dfy,dfx)
    print (lrmodel.summary())
#if __name__ == "__LRMLE__":
    #LRMLE()
 #---------------------------------------------       
def l1():
    #load data
    dfy, dfx = load_data(s.dataset)
    #L1 error
    sumofloss = lr.l1(dfy,dfx)
    print(sumofloss.popularity)
    
#if __name__ == "__l1__":
   # l1()
#---------------------------------------------    
def bootme():
    #load
    dfy,dfx = load_data(s.dataset)
    #bootstrap
    coefboot = bt.boot(dfy,dfx)
    print(coefboot )

#if __name__ == "__bootme__":
   # bootme()
    
def plotCI():
    #load
    dfy,dfx = load_data(s.dataset)
    #bootstrap
    coefboot = bt.boot(dfy,dfx)
    print(coefboot )

#if __name__ == "__plotCI__":
    #plotCI()
#--------------------------------------------- 
def clust():
        true_labels, df = load_data(s.dataset) 
        pred_labels = kmeans_cluster(df, s.n_components)
        classification_measure(true_labels, pred_labels)
#if __name__ == "__clust__":
    #clust()    
    
#__________________________________________________________   

if __name__ == "__main__":
    LRMLE()
    l1()
    bootme()
    plotCI()
    clust()
    
    
    
    
    
    
    
    
    
    
    
    
    
    