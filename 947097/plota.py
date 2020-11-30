#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:22:29 2020

@author: Jyanqa
"""


import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def lineplot_with_errors(data, bounds, name="data"):
    
    x = list(range(len(data)))
    
    fig = go.Figure(
        [
            go.Scatter(x=x,
                       y=data,
                       mode='lines',
                       name=name,
                       line={"color": "red"}
                ),
            go.Scatter(x=x + x[::-1],
                       y=bounds[0] + bounds[1][::-1],
                       fill='toself',
                       hoverinfo='skip',
                       showlegend=False,
                       line={"color": "orange"}
                )
        ]
    )
    
    fig.show()
    
def scatterplot_with_colors(data, colors, **kwargs):
    if data.shape[1] > 2:
        data = StandardScaler().fit_transform(data)
        pca = PCA(n_components = 2).fit(data)
        explained_variance = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_variance:.2f}")
        data = pca.transform(data)
    fig = px.scatter(x=data[:, 0], y=data[:, 1], color=colors, **kwargs)
    fig.show()
    
    

    
    
    
    
    
    