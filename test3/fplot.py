import random
import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import signal
import pickle
import glob

files = glob.glob("*.pickle")

unpacked = []


for x in files:
    if x.startswith("IGN"):
        files.remove(x)
        print("REMOVED: " + str(x))
        continue


for x in files:
    with open(x, 'rb') as handle:
        unpacked.append(pickle.load(handle))

mini = min(len(x) for x in unpacked)
for x in range(len(unpacked)):
    unpacked[x] = unpacked[x][:mini].T


exclude = {}
groups = {'RealDRQN': 'orange', 'Random': 'black', "DQN": 'green', 'Tanh':'blue'}

legenda = []

alel = ['SCORE', 'MAX SCORE', 'AVG SCORE', 'AVG MAX SCORE']
def printPos0(unpackeds,files,lvl):
    grpCount = {}
    plt.clf()
    texts = []
    from adjustText import adjust_text
    for idx, x in enumerate(unpackeds):
        y1 = x[lvl]
        col = 'red'
        #print(files[0])
        contin = True
        for tipes in exclude:
            if files[idx].endswith( tipes):
                contin = False
                break
        if contin == False:
            continue
        
        legenda.append(files[idx])
        for tipe in groups:
            if files[idx].endswith( tipe):
                if tipe in grpCount:
                    grpCount[tipe] += 1
                else:
                    grpCount[tipe] = 1
                col = groups[tipe]
        
        
        plt.plot(y1,color=col, alpha=0.5)
        maxSofar = 0
        for xy in zip(x[4],y1):
            if xy[1] > maxSofar:
                maxSofar = xy[1]

    plt.title('Assault - Atari')
    plt.ylabel(alel[lvl])
    plt.xlabel('episode')

    plt.legend(legenda, bbox_to_anchor=(1.04,1), borderaxespad=0)

    if lvl == 2:
        plt.gca().set_ylim(top=1100, bottom=500)
    plt.savefig('compa'+str(lvl)+'.png', bbox_inches="tight")
    print(grpCount)


for lvl in range(0,4):
    files = [x.replace(".pickle", "") for x in files]
    printPos0(unpacked, files, lvl)

