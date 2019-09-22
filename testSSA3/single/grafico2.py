#%% [markdown]
# Inicialização

#%%
import random
import math
import numpy as np
from collections import deque
#import matplotlib
#import matplotlib.pyplot as plt
import signal
import pickle
import glob
import pandas as pd
#from mpl_toolkits.mplot3d import Axes3D  

#import plotly.plotly as py
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

plotly.offline.init_notebook_mode(connected=True)


#%%
def normalize(list, range): # range should be (lower_bound, upper_bound)
    l = np.array(list) 
    a = np.max(l)
    c = np.min(l)
    b = range[1]
    d = range[0]

    m = (b - d) / (a - c)
    pslope = (m * (l - c)) + d
    return pslope

def loadFiles(loadgroups):
    files = glob.glob("*.pickle")
    files = [x.replace(".pickle", "") for x in files]
    unpacked = []
    for x in files:
        for acp in loadgroups:
            if x.endswith(acp):
                with open(x+".pickle", 'rb') as handle:
                    unpacked.append([x,pickle.load(handle)[0:]])
                break
    return unpacked


#%%
groups = {  'testForgetting2' : 'blue'
            }
loadedfile = loadFiles(groups)
columnsofdt = ['SCORE', 'MAX SCORE', 'AVG SCORE', 'AVG MAX SCORE', 'EPISODE']

processed = {}

def getLabelName(inp, dictx):
    for x in dictx:
        if inp.endswith(x):
            return x
    return None



for runtst in loadedfile:
    posIn = getLabelName(runtst[0], groups)
    if posIn in processed:
        processed[posIn] = np.average([processed[posIn], runtst[1]], axis=0)
    else:
        processed[posIn] = runtst[1]



count = 0


#%%
def calcAvgOfXWStrides(scores, avgx, strides):
    y1 = []

    for pos in range(0, len(scores), strides):
        y1.append(np.average(scores[pos:pos+avgx]))

    return y1



def addToPlotAvgOfXWStrides(scores, avg, labelx, col, strides, posx=1, maxpos=1, padding=0.3, title=""):
    y1 = calcAvgOfXWStrides(scores, avg, strides)
    y2 = calcAvgOfXWStrides(scores, 100, 1)
    
    
    x = np.array([i for i in range(0, len(y1))])
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)

    blueshift = [0 for x in range(0,3620)]
    blueshift.extend([150 for x in range(3620,6570)])
    blueshift.extend([255 for x in range(6570,len(y1))])

    
    diffd = np.subtract(y1,y2)

    ptp = np.ptp(diffd).astype(int)
    internal = (diffd - np.min(diffd))
    l_norm =  255*np.divide(internal,ptp)


    colors = ["rgba(10, "+str(abs(l_norm[x]))+", "+str(blueshift[x])+", 1)" if y1[x] > y2[x] else "rgba(10, 10, "+str(blueshift[x])+", 1)" if y1[x] == y2[x] else "rgba("+str(abs(l_norm[x]))+", 10, "+str(blueshift[x])+", 1)"  for x in range(0,len(y1))]

    trace1 = go.Scatter(
        x=posit,
        y=y1,
        mode='markers',
        marker=dict(
            size=6, 
            color=colors,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=pospad
            ),
            opacity=0.8
        ),
        name=title
    )
    data = [trace1]
    layout = go.Layout(
        title=go.layout.Title(
        text=title+' - Average of ' + str(avg),
        xref='paper',
        x=0
        ),
        scene = dict(
            
                    xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="rgb(255, 255, 255)",
                         showbackground=True,
                         zerolinecolor="rgb(255, 255, 255)",
                         title='Scores'),
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="rgb(255, 255, 255)",
                        showbackground=True,
                        zerolinecolor="rgb(255, 255, 255)",
                        title='Difference'),)
                    
                  
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename=title+' CAVGoF'+str(avg)+'-'+str(random.randint(1,50000)))

    
def plotAndSaveContinuousAverageOf(customTitle,title,averageOf = 1,blim=None, ulim=None,customLabel = None, posx = 1, maxn = 1):
    for runtst in processed:
        dtframe = processed[runtst]
        label = runtst
        if customLabel is not None:
            label=customLabel
        color = groups[runtst]

        print("Processing: " + label)
        dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
        scores = dataframe['SCORE']
        
        addToPlotAvgOfXWStrides(scores, averageOf, label, color, 1, title=title)
    print("Done")
    
blim = None
ulim = None
title = 'Assault - Atari'
ctitle = 'Forgetting'   



#%%
plotAndSaveContinuousAverageOf(ctitle,title,1,blim,ulim,ctitle)


#%%
plotAndSaveContinuousAverageOf(ctitle,title,10,blim,ulim,ctitle)


#%%
plotAndSaveContinuousAverageOf(ctitle,title,100,blim,ulim,ctitle)


#%%

#%%



#%%



