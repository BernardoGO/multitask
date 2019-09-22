import random
import gym
import math
import numpy as np
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import signal
import pickle
import glob
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  

#import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#matplotlib.use('cairo') 
#Nos primeros 10, sem agrupar, cada grafico com 1 ou 1vsrandom
#Maior e menor ponto, desvio padrão, media

ax = None


def addToPlotDiffCent(scores_, labelx, avg, col, posx=1, maxpos=1, padding=0.3, title=""):
    scores = calcAvgOfX(scores_, avg)
    y1 = []

    for pos in range(1, len(scores)-1, 1):
        y1.append((scores[pos+1] - scores[pos-1])/2)

    x = np.array([i for i in range(0, len(y1))])
    #print(1/maxpos)
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)
    plt.bar(posit,y1,color=col, alpha=0.5,label=labelx, width=pospad)
    

    


    trace1 = go.Scatter(
        x=posit,
        y=y1,
        mode='markers',
        marker=dict(
            size=6,
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
    #py.plot(fig, filename=title+'-avgOf' + str(avg)+'-'+str(random.randint(1,50000)))
    # Plot and embed in ipython ndataotebook!
    plot_url = plot(fig, filename=title+'DiffCent'+str(random.randint(1,50000)))


    
    

def countPosNeg(y1):
    negatives = sum(n < 0 for n in y1)
    positives = sum(n > 0 for n in y1)
    zeroes = sum(n == 0 for n in y1)
    return [negatives, zeroes, positives]

def countWeightedPosNeg(y1):
    negatives = sum(n if (n <  0) else 0 for n in y1)
    positives = sum(n if (n >  0) else 0 for n in y1)
    zeroes =    sum(n if (n == 0) else 0 for n in y1)
    return [negatives, zeroes, positives]

def genDiffProg(scores_, avg, labelx, col, posx=1, maxpos=1, padding=0.3):
    scores = calcAvgOfX(scores_, avg)
    y1 = [0]

    for pos in range(0, len(scores)-1, 1):
        y1.append((scores[pos+1] - scores[pos]))

    #x = [(maxpos*i)+posx for i in range(0, len(y1))]
    return y1





def addToPlotDiffProg(scores_, avg, labelx, col, posx=1, maxpos=1, padding=0.3, title=""):
    y1 = genDiffProg(scores_,avg, labelx, col, posx, maxpos, padding)

    print(countPosNeg(y1))
    print(countWeightedPosNeg(y1))
    x = np.array([i for i in range(0, len(y1))])
    #print(1/maxpos)
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)
    plt.bar(posit,y1,color=col, alpha=0.5,label=labelx, width=pospad)





    #plt.plot(y1,color=col, alpha=0.5,label=labelx)
    trace1 = go.Bar(
        x=posit,
        y=y1,
        marker=dict(
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
    """,
                    width=700,
                    margin=dict(
                    r=20, b=10,
                    l=10, t=10)
                    """
    fig = go.Figure(data=data, layout=layout)
    #py.plot(fig, filename=title+'-avgOf' + str(avg)+'-'+str(random.randint(1,50000)))
    # Plot and embed in ipython notebook!
    #plot_url = plot(data, filename=title+'DiffCent'+str(random.randint(1,50000)))

    # Plot and embed in ipython notebook!
    plot_url = plot(fig, filename=title+'DiffProg'+str(random.randint(1,50000)))

    # or plot with: plot_url = plot(data, filename='basic-line')




def addToPlotDiffAvgWProg(scores_, avg, labelx, col, posx=1, maxpos=1, padding=0.3, title=""):
    
    y1 = genDiffProg(scores_,avg, labelx, col, posx, maxpos, padding)
    scores = calcAvgOfX(scores_, avg)

    

    print(countPosNeg(y1))
    print(countWeightedPosNeg(y1))
    x = np.array([i for i in range(0, len(y1))])
    #print(1/maxpos)
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)

    print(len(y1))
    print(len(scores))
    print(len(posit))
    
    

   
    
    #ax.plot_trisurf(scores,y1,posit)
    ax.scatter(scores,y1,posit, c="r", s=pospad)
    #plt.scatter(scores,y1,color=col, alpha=0.5,label=labelx, s=pospad)
    #plt.plot(y1,color=col, alpha=0.5,label=labelx)

    trace1 = go.Scatter3d(
        x=scores,
        y=y1,
        z=posit,
        mode='markers',
        marker=dict(
            size=6,
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
                        title='Difference'),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="rgb(255, 255, 255)",
                        showbackground=True,
                        zerolinecolor="rgb(255, 255, 255)",
                        title='Step'),)
                        
                    
                  
    )
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename=title+'-avgOf' + str(avg)+'-'+str(random.randint(1,50000)))


def addToPlotMinMax(scores_, avg, labelx, col, posx=1, maxpos=1, padding=0.3, title=""):
    
    #y1 = genDiffProg(scores_,avg, labelx, col, posx, maxpos, padding)
    #avgMinMax = calcAvgOfXMinMax(scores_, avg)

    y1 = []
    #scores[pos:pos+avgx]
    #for pos in range(0, len(scores)-1, 1):
    #    y1.append((scores[pos+1] - scores[pos]))
    for pos in range(0, len(scores_), avg):
        y1.append((np.amax(scores_[pos:pos+avg]) - np.amin(scores_[pos:pos+avg]))/np.average(scores_[pos:pos+avg]))
    #print(countPosNeg(y1))
    #print(countWeightedPosNeg(y1))
    x = np.array([i for i in range(0, len(y1))])
    #print(1/maxpos)
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)

    print(len(y1))
    #print(len(scores))
    print(len(posit))
    plt.scatter(posit, y1, c="r", s=pospad)
    

"""
def addToPlotDiffProg2nd(scores_, labelx, col, posx=1, maxpos=1, padding=0.3):
    scores = calcAvgOfX(scores_, 100)
    y1 = []

    for pos in range(0, len(scores)-2, 1):
        y1.append(scores[pos+2] - 2*scores[pos+1] + scores[pos])

    #x = [(i/maxpos)*posx for i in range(0, len(y1))]
    x = np.array([i for i in range(0, len(y1))])
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)
    plt.bar(posit,y1,color=col, alpha=0.5,label=labelx, width=pospad)
    #plt.plot(y1,color=col, alpha=0.5,label=labelx)
"""

def genDiffDiffProg2nd(scores_, avg, labelx, col, posx=1, maxpos=1, padding=0.3, title=""):
    scores = calcAvgOfX(scores_, avg)
    y1 = []

    for pos in range(0, len(scores)-2, 1):
        y1.append(scores[pos+2] - 2*scores[pos+1] + scores[pos])


    #x = [(maxpos*i)+posx for i in range(0, len(y1))]
    return y1

def addToPlotDiffProg2nd(scores_, avg, labelx, col, posx=1, maxpos=1, padding=0.3, title=""):
    y1 = genDiffDiffProg2nd(scores_,avg, labelx, col, posx, maxpos, padding)

    print(countPosNeg(y1))
    print(countWeightedPosNeg(y1))
    x = np.array([i for i in range(0, len(y1))])
    #print(1/maxpos)
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)
    plt.bar(posit,y1,color=col, alpha=0.5,label=labelx, width=pospad)

def genDiffDiffCent2nd(scores_, avg, labelx, col, posx=1, maxpos=1, padding=0.3):
    scores = calcAvgOfX(scores_, avg)
    y1 = []

    for pos in range(1, len(scores)-1, 1):
        y1.append((scores[pos+1] - 2*scores[pos] + scores[pos-1])/4)

    #x = [(maxpos*i)+posx for i in range(0, len(y1))]
    return y1



def addToPlotDiffCent2nd(scores_, avg, labelx, col, posx=1, maxpos=1, padding=0.3, title=""):
    y1 = genDiffDiffCent2nd(scores_,avg, labelx, col, posx, maxpos, padding)

    print(countPosNeg(y1))
    print(countWeightedPosNeg(y1))
    x = np.array([i for i in range(0, len(y1))])
    #print(1/maxpos)
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)
    plt.bar(posit,y1,color=col, alpha=0.5,label=labelx, width=pospad)

def addToPlotDiffReg2nd(scores_, labelx, col, posx=1, maxpos=1, title=""):
    scores = calcAvgOfX(scores_, 10)
    y1 = []

    for pos in range(2, len(scores), 1):
        y1.append(scores[pos] - 2*scores[pos-1] + scores[pos-2])

    x = [(i/maxpos)*posx for i in range(0, len(y1))]

    plt.bar(x,y1,color=col, alpha=0.5,label=labelx)

def addToPlotDiffReg(scores_, labelx, col, posx=1, maxpos=1, padding=0.3, title=""):
    scores = calcAvgOfX(scores_, 100)
    y1 = []

    for pos in range(1, len(scores), 1):
        y1.append(scores[pos] - scores[pos-1])

    x = np.array([i for i in range(0, len(y1))])
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)
    plt.bar(posit,y1,color=col, alpha=0.5,label=labelx, width=pospad)


def calcAvgOfX(scores, avgx):
    y1 = []

    for pos in range(0, len(scores), avgx):
        y1.append(np.average(scores[pos:pos+avgx]))

    return y1

def calcAvgOfXMinMax(scores, avgx):
    y1 = []
    min_ = []
    max_ = []

    for pos in range(0, len(scores), avgx):
        y1.append(np.average(scores[pos:pos+avgx]))
        min_.append(np.amin(scores[pos:pos+avgx]))
        max_.append(np.amax(scores[pos:pos+avgx]))

    return [y1,min_,max_]




def addToPlotAvgOfX(scores, avg, labelx, col, posx=1, maxpos=1, padding=0.3, title=""):
    y1 = calcAvgOfX(scores, avg)
    
    #plt.plot(y1,color=col, alpha=0.5,label=labelx)
    x = np.array([i for i in range(0, len(y1))])
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)
    xs = np.linspace(0, len(y1), len(y1))
    plt.scatter(posit,y1,color=col, alpha=0.5,label=labelx,s=pospad)

    
    
    colors = ["rgba(10, 200, 10, 1)" for x in range(0,3620//avg)]
    colors.extend(["rgba(200, 200, 10, 1)" for x in range(3620//avg,6570//avg)])
    colors.extend(["rgba(200, 10, 10, 1)" for x in range(6570//avg,len(y1))])
    
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
    #py.plot(fig, filename=title+'-avgOf' + str(avg)+'-'+str(random.randint(1,50000)))
    # Plot and embed in ipython notebook!
    #plot_url = plot(data, filename=title+'DiffCent'+str(random.randint(1,50000)))
    # Plot and embed in ipython notebook!
    plot_url = plot(fig, filename=title+' AvgOf' + str(avg)+'-'+str(random.randint(1,50000)))
    #plt.gca().set_ylim(top=1100, bottom=500)

def calcAvgOfXWStrides(scores, avgx, strides):
    y1 = []

    for pos in range(0, len(scores), strides):
        y1.append(np.average(scores[pos:pos+avgx]))

    return y1

def normalize(list, range): # range should be (lower_bound, upper_bound)
    l = np.array(list) 
    a = np.max(l)
    c = np.min(l)
    b = range[1]
    d = range[0]

    m = (b - d) / (a - c)
    pslope = (m * (l - c)) + d
    return pslope

def addToPlotAvgOfXWStrides(scores, avg, labelx, col, strides, posx=1, maxpos=1, padding=0.3, title=""):
    y1 = calcAvgOfXWStrides(scores, avg, strides)

    #plt.plot(y1,color=col, alpha=0.5,label=labelx)
    
    x = np.array([i for i in range(0, len(y1))])
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)
    xs = np.linspace(0, len(y1), len(y1))
    plt.scatter(posit,y1,color=col, alpha=0.5,label=labelx,s=pospad)
    #colors = ["rgba(10, 200, 10, 1)" for x in range(0,3620//1)]
    #colors.extend(["rgba(200, 200, 10, 1)" for x in range(3620//1,6570//1)])
    #colors.extend(["rgba(200, 10, 10, 1)" for x in range(6570//1,len(y1))])

    blueshift = [0 for x in range(0,3620//1)]
    blueshift.extend([150 for x in range(3620//1,6570//1)])
    blueshift.extend([255 for x in range(6570//1,len(y1))])

    y2 = calcAvgOfXWStrides(scores, 100, 1)
    diffd = np.subtract(y1,y2)
    diffmax = np.max(diffd)
    diffmin = np.min(diffd)
    diffd = np.add(diffd, abs(diffmin)+1)
    #lower, upper = 10, 240
    #l_norm = [lower + (upper - lower) * x for x in diffd]
    #10-240
    l_norm =  255*(diffd - np.min(diffd))/np.ptp(diffd).astype(int)

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
    #py.plot(fig, filename=title+'-avgOf' + str(avg)+'-'+str(random.randint(1,50000)))
    # Plot and embed in ipython notebook!
    #plot_url = plot(data, filename=title+'DiffCent'+str(random.randint(1,50000)))

    # Plot and embed in ipython notebook!
    plot_url = plot(fig, filename=title+' CAVGoF'+str(avg)+'-'+str(random.randint(1,50000)))
    #plt.gca().set_ylim(top=1100, bottom=500)

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
    

def plotScoresAll(dtframe, label, color, posx = 1, maxn = 1):
    print("Processing: " + label)
    dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
    scores = dataframe['SCORE']
    #addToPlotAvgOfX(scores, 100, label, color)
    addToPlotDiffProg(scores, 100, label, color, posx, maxn)
    #addToPlotAvgOfXWStrides(scores, 10, label, color, 1)



groups = {  'testForgetting2' : 'blue'
            }

"""
'DQN' : 'blue',
            'DRQN': 'orange',
            
            'Random': 'black', 
            "DQN": 'green', 
            'Tanh':'blue'
"""


# diferença progressiva
# diferença regressiva
# diferença centrada
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
#for runtst in processed:
    
#    plotScoresAll(processed[runtst], runtst, groups[runtst], count, len(processed))
#    count+=1


#dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
#scores = dataframe['SCORE']




def plotAndSaveForwardDifference(customTitle,title,averageOf = 1,blim=None, ulim=None,customLabel = None, posx = 1, maxn = 1):
    plt.clf()
    for runtst in processed:
        dtframe = processed[runtst]
        label = runtst
        if customLabel is not None:
            label=customLabel
        color = groups[runtst]

        print("Processing: " + label)
        dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
        scores = dataframe['SCORE']
        
        addToPlotDiffProg(scores, averageOf, label, color, posx, maxn, title=title)
        #addToPlotAvgOfX(scores, 100, label, color, title=title)
        #addToPlotAvgOfXWStrides(scores, 10, label, color, 1, title=title)
    plt.title(title)
    #plt.ylabel("Diferença Progressiva")
    plt.ylabel("Reward - Forward Difference - Average of " + str(averageOf))
    plt.xlabel('Step')
    plt.legend(loc='upper right')
    #plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    if blim is not None and ulim is not None:
        plt.gca().set_ylim(top=ulim, bottom=blim)
    plt.savefig(customTitle+'-FWD-'+str(averageOf)+'.png', bbox_inches="tight")

    print("Saved!")



def plotAndSaveCentral2ndDifference(customTitle,title,averageOf = 1,blim=None, ulim=None,customLabel = None, posx = 1, maxn = 1):
    plt.clf()
    for runtst in processed:
        dtframe = processed[runtst]
        
        label = runtst
        if customLabel is not None:
            label=customLabel

        color = groups[runtst]

        print("Processing: " + label)
        dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
        scores = dataframe['SCORE']
        
        addToPlotDiffCent2nd(scores, averageOf, label, color, posx, maxn, title=title)
        #addToPlotAvgOfX(scores, 100, label, color)
        #addToPlotAvgOfXWStrides(scores, 10, label, color, 1)
    plt.title(title)
    #plt.ylabel("Diferença Progressiva")
    plt.ylabel("Reward - 2nd Cent. Difference - Average of " + str(averageOf))
    plt.xlabel('Step')
    plt.legend(loc='upper right')
    if blim is not None and ulim is not None:
        plt.gca().set_ylim(top=ulim, bottom=blim)
    #plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.savefig(customTitle+'-2ndCNT-'+str(averageOf)+'.png', bbox_inches="tight")

    print("Saved!")

def plotAndSaveFwd2ndDifference(customTitle,title,averageOf = 1,blim=None, ulim=None,customLabel = None, posx = 1, maxn = 1):
    plt.clf()
    for runtst in processed:
        dtframe = processed[runtst]
        
        label = runtst
        if customLabel is not None:
            label=customLabel

        color = groups[runtst]

        print("Processing: " + label)
        dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
        scores = dataframe['SCORE']
        
        addToPlotDiffCent2nd(scores, averageOf, label, color, posx, maxn, title=title)
        #addToPlotAvgOfX(scores, 100, label, color)
        #addToPlotAvgOfXWStrides(scores, 10, label, color, 1)
    plt.title(title)
    #plt.ylabel("Diferença Progressiva")
    plt.ylabel("Reward - 2nd Forw. Difference - Average of " + str(averageOf))
    plt.xlabel('Step')
    plt.legend(loc='upper right')
    if blim is not None and ulim is not None:
        plt.gca().set_ylim(top=ulim, bottom=blim)
    #plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.savefig(customTitle+'-2ndFWD-'+str(averageOf)+'.png', bbox_inches="tight")

    print("Saved!")


#addToPlotDiffAvgWProg


def plotAndSaveHib(customTitle,title,averageOf = 1,blim=None, ulim=None,customLabel = None, posx = 1, maxn = 1):
    global ax
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for runtst in processed:
        dtframe = processed[runtst]
        label = runtst
        if customLabel is not None:
            label=customLabel
        color = groups[runtst]

        print("Processing: " + label)
        dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
        scores = dataframe['SCORE']
        
        #addToPlotDiffProg(scores, averageOf, label, color, posx, maxn)
        addToPlotDiffAvgWProg(scores, averageOf, label, color, title=title)
        #addToPlotAvgOfXWStrides(scores, averageOf, label, color, 1)
    plt.title(title + "- Average of " + str(averageOf))
    #plt.ylabel("Diferença Progressiva")
    
    ax.set_xlabel('Scores')
    ax.set_ylabel("Difference")
    ax.set_zlabel("Step")
    plt.legend(loc='upper right')
    #plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    if blim is not None and ulim is not None:
        plt.gca().set_ylim(top=ulim, bottom=blim)
    #plt.show()
    plt.savefig(customTitle+'-HIB-'+str(averageOf)+'.png', bbox_inches="tight")

    print("Saved!")
#
def plotAndSaveAverageOf(customTitle,title,averageOf = 1,blim=None, ulim=None,customLabel = None, posx = 1, maxn = 1):
    plt.clf()
    for runtst in processed:
        dtframe = processed[runtst]
        label = runtst
        if customLabel is not None:
            label=customLabel
        color = groups[runtst]

        print("Processing: " + label)
        dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
        scores = dataframe['SCORE']
        
        #addToPlotDiffProg(scores, averageOf, label, color, posx, maxn, title=title)
        addToPlotAvgOfX(scores, averageOf, label, color, title=title)
        #addToPlotAvgOfXWStrides(scores, averageOf, label, color, 1, title=title)
    plt.title(title)
    #plt.ylabel("Diferença Progressiva")
    plt.ylabel("Reward - Average of " + str(averageOf))
    plt.xlabel('Step')
    plt.legend(loc='upper right')
    #plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    if blim is not None and ulim is not None:
        plt.gca().set_ylim(top=ulim, bottom=blim)
    #plt.show()
    plt.savefig(customTitle+'-AVG-'+str(averageOf)+'.png', bbox_inches="tight")

    print("Saved!")

def plotAndSaveAvgMinMaxOf(customTitle,title,averageOf = 1,blim=None, ulim=None,customLabel = None, posx = 1, maxn = 1):
    plt.clf()
    for runtst in processed:
        dtframe = processed[runtst]
        label = runtst
        if customLabel is not None:
            label=customLabel
        color = groups[runtst]

        print("Processing: " + label)
        dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
        scores = dataframe['SCORE']
        
        #addToPlotDiffProg(scores, averageOf, label, color, posx, maxn)
        addToPlotMinMax(scores, averageOf, label, color, title=title)
        #addToPlotAvgOfXWStrides(scores, averageOf, label, color, 1)
    plt.title(title)
    #plt.ylabel("Diferença Progressiva")
    plt.ylabel("minmax - Average of " + str(averageOf))
    plt.xlabel('Step')
    plt.legend(loc='upper right')
    #plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    if blim is not None and ulim is not None:
        plt.gca().set_ylim(top=ulim, bottom=blim)
    #plt.show()
    plt.savefig(customTitle+'-MInMAX-'+str(averageOf)+'.png', bbox_inches="tight")

    print("Saved!")

def plotAndSaveContinuousAverageOf(customTitle,title,averageOf = 1,blim=None, ulim=None,customLabel = None, posx = 1, maxn = 1):
    plt.clf()
    for runtst in processed:
        dtframe = processed[runtst]
        label = runtst
        if customLabel is not None:
            label=customLabel
        color = groups[runtst]

        print("Processing: " + label)
        dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
        scores = dataframe['SCORE']
        
        #addToPlotDiffProg(scores, averageOf, label, color, posx, maxn)
        #addToPlotAvgOfX(scores, averageOf, label, color)
        addToPlotAvgOfXWStrides(scores, averageOf, label, color, 1, title=title)
    plt.title(title)
    #plt.ylabel("Diferença Progressiva")
    plt.ylabel("Reward - Average of " + str(averageOf))
    plt.xlabel('Step')
    plt.legend(loc='upper right')
    if blim is not None and ulim is not None:
        plt.gca().set_ylim(top=ulim, bottom=blim)
    #plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.savefig(customTitle+'-CAVG-'+str(averageOf)+'.png', bbox_inches="tight")

    print("Saved!")

blim = None
ulim = None
title = 'Assault - Atari'
ctitle = 'Forgetting'

#plotAndSaveHib(ctitle,title,1,blim,ulim,ctitle)
#plotAndSaveHib(ctitle,title,10,blim,ulim,ctitle)

#plotAndSaveForwardDifference(ctitle,title,10,blim,ulim,ctitle)

plotAndSaveForwardDifference(ctitle,title,1,blim,ulim,ctitle)
plotAndSaveForwardDifference(ctitle,title,10,blim,ulim,ctitle)
plotAndSaveForwardDifference(ctitle,title,100,blim,ulim,ctitle)
#plotAndSaveCentral2ndDifference(ctitle,title,1,blim,ulim,ctitle)
#plotAndSaveCentral2ndDifference(ctitle,title,10,blim,ulim,ctitle)
#plotAndSaveCentral2ndDifference(ctitle,title,100,blim,ulim,ctitle)

plotAndSaveFwd2ndDifference(ctitle,title,1,blim,ulim,ctitle)
plotAndSaveFwd2ndDifference(ctitle,title,10,blim,ulim,ctitle)
plotAndSaveFwd2ndDifference(ctitle,title,100,blim,ulim,ctitle)

blim = 0
ulim = 5000
#plotAndSaveAverageOf(ctitle,title,1,blim,ulim,ctitle)
#plotAndSaveAverageOf(ctitle,title,10,blim,ulim,ctitle)
#plotAndSaveAverageOf(ctitle,title,100,blim,ulim,ctitle)

#plotAndSaveContinuousAverageOf(ctitle,title,1,blim,ulim,ctitle)
#plotAndSaveContinuousAverageOf(ctitle,title,10,blim,ulim,ctitle)
#plotAndSaveContinuousAverageOf(ctitle,title,100,blim,ulim,ctitle)

#blim = -1850
#ulim = +1850

#plotAndSaveHib(ctitle,title,1,blim,ulim,ctitle)
#plotAndSaveHib(ctitle,title,10,blim,ulim,ctitle)
#plotAndSaveHib(ctitle,title,100,blim,ulim,ctitle)
plotAndSaveAvgMinMaxOf(ctitle,title,1,blim,ulim,ctitle)
plotAndSaveAvgMinMaxOf(ctitle,title,10,blim,ulim,ctitle)
plotAndSaveAvgMinMaxOf(ctitle,title,100,blim,ulim,ctitle)


#val[i]