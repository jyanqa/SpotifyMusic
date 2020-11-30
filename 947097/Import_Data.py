#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:33:13 2020

@author: Jyanqa
"""
import pandas as pd
import Setting as s
#import re

def load_data(filename: str):
     df = pd.read_csv(filename)
     df = df.dropna()
     #run function with the first 500 rows
     df = df.iloc[1:1000] 
     dfy = df[s.y]
     dfx = df[s.x]
     return dfy, dfx

