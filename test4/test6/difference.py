import random
import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import signal
import pickle
import glob
import pandas as pd

#Nos primeros 10, sem agrupar, cada grafico com 1 ou 1vsrandom
#Maior e menor ponto, desvio padrão, media

def addToPlotDiffCent(scores_, labelx, col, posx=1, maxpos=1):
    scores = calcAvgOfX(scores_, 10)
    y1 = []

    for pos in range(1, len(scores)-1, 1):
        y1.append((scores[pos+1] - scores[pos-1])/2)

    x = [(i/maxpos)*posx for i in range(0, len(y1))]

    plt.bar(x,y1,color=col, alpha=0.5,label=labelx)


def addToPlotDiffProg(scores_, labelx, col, posx=1, maxpos=1, padding=0.3):
    scores = calcAvgOfX(scores_, 10)
    y1 = []

    for pos in range(0, len(scores)-1, 1):
        y1.append(scores[pos+1] - scores[pos])

    #x = [(maxpos*i)+posx for i in range(0, len(y1))]
    x = np.array([i for i in range(0, len(y1))])
    #print(1/maxpos)
    pospad = (1/maxpos)-padding/maxpos
    posit = (x-pospad)+(posx*pospad)
    plt.bar(posit,y1,color=col, alpha=0.5,label=labelx, width=pospad)

def addToPlotDiffProg2nd(scores_, labelx, col, posx=1, maxpos=1):
    scores = calcAvgOfX(scores_, 10)
    y1 = []

    for pos in range(0, len(scores)-2, 1):
        y1.append(scores[pos+2] - 2*scores[pos+1] + scores[pos])

    x = [(i/maxpos)*posx for i in range(0, len(y1))]

    plt.bar(x,y1,color=col, alpha=0.5,label=labelx)


def addToPlotDiffCent2nd(scores_, labelx, col, posx=1, maxpos=1):
    scores = calcAvgOfX(scores_, 10)
    y1 = []

    for pos in range(1, len(scores)-1, 1):
        y1.append((scores[pos+1] - 2*scores[pos] + scores[pos-1])/4)

    x = [(i/maxpos)*posx for i in range(0, len(y1))]

    plt.bar(x,y1,color=col, alpha=0.5,label=labelx)

def addToPlotDiffReg2nd(scores_, labelx, col, posx=1, maxpos=1):
    scores = calcAvgOfX(scores_, 10)
    y1 = []

    for pos in range(2, len(scores), 1):
        y1.append(scores[pos] - 2*scores[pos-1] + scores[pos-2])

    x = [(i/maxpos)*posx for i in range(0, len(y1))]

    plt.bar(x,y1,color=col, alpha=0.5,label=labelx)

def addToPlotDiffReg(scores_, labelx, col, posx=1, maxpos=1):
    scores = calcAvgOfX(scores_, 10)
    y1 = []

    for pos in range(1, len(scores), 1):
        y1.append(scores[pos] - scores[pos-1])

    x = [(i/maxpos)*posx for i in range(0, len(y1))]

    plt.bar(x,y1,color=col, alpha=0.5,label=labelx)


def calcAvgOfX(scores, avgx):
    y1 = []

    for pos in range(0, len(scores), avgx):
        y1.append(np.average(scores[pos:pos+avgx]))

    return y1


def addToPlotAvgOfX(scores, avgx, labelx, col):
    y1 = calcAvgOfX(scores, avgx)

    plt.plot(y1,color=col, alpha=0.5,label=labelx)
    
    #plt.gca().set_ylim(top=1100, bottom=500)

def loadFiles(loadgroups):
    files = glob.glob("*.pickle")
    files = [x.replace(".pickle", "") for x in files]
    unpacked = []
    for x in files:
        for acp in loadgroups:
            if x.endswith(acp):
                with open(x+".pickle", 'rb') as handle:
                    unpacked.append([x,pickle.load(handle)])
                break
    return unpacked
    

def plotScoresAll(dtframe, label, color, posx = 1, maxn = 1):
    dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
    scores = dataframe['SCORE']
    addToPlotAvgOfX(scores, 10, label, color)
    addToPlotDiffProg(scores, label, color, posx, maxn)

groups = {  'DQN': 'orange',
            'RealDRQN' : 'red',
            'Random': 'black'}

"""
, 
            'Random': 'black', 
            "DQN": 'green', 
            'Tanh':'blue'
"""


# diferença progressiva
# diferença regressiva
# diferença centrada
loadedfile = loadFiles(groups)
columnsofdt = ['SCORE', 'MAX SCORE', 'AVG SCORE', 'AVG MAX SCORE', 'EPISODE']

plt.clf()

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
for runtst in processed:
    
    plotScoresAll(processed[runtst], runtst, groups[runtst], count, len(processed))
    count+=1

#plt.gca().set_ylim(top=1100, bottom=500)

plt.title('Assault - Atari')
plt.ylabel("Diferença Progressiva")
plt.xlabel('episode')
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
plt.savefig('compa8'+'.png', bbox_inches="tight")

print("Loaded")

