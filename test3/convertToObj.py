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



def addToPlotAvgOfX(scores, avgx, labelx, col):
    y1 = []

    for pos in range(0, len(scores), avgx):
        y1.append(np.average(scores[pos:pos+avgx]))

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
    

def plotScoresAll(dtframe, label, color):
    dataframe = pd.DataFrame(data=dtframe, columns=columnsofdt)
    scores = dataframe['SCORE']
    addToPlotAvgOfX(scores, 10, label, color)

groups = {  'RealDRQN': 'orange', 
            'Random': 'black', 
            "DQN": 'green', 
            'Tanh':'blue'}



loadedfile = loadFiles(groups)
columnsofdt = ['SCORE', 'MAX SCORE', 'AVG SCORE', 'AVG MAX SCORE', 'EPISODE']

plt.clf()

for runtst in loadedfile:
    color = 'red'
    for x in groups:
        if runtst[0].endswith(x):
            color = groups[x]
    plotScoresAll(runtst[1], runtst[0], color)


plt.title('Assault - Atari')
plt.ylabel("labl")
plt.xlabel('episode')
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
plt.savefig('compa6'+'.png', bbox_inches="tight")

print("Loaded")

