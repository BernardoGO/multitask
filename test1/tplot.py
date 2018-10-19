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

# falsee = "using-False.pickle"
# falsee256 = "using-False256-2.pickle"
# truee = "using-True.pickle"
# h_falsee = None
# h_falsee256 = None
# h_truee = None
# with open(falsee256, 'rb') as handle:
#     h_falsee256 = pickle.load(handle)
# with open(falsee, 'rb') as handle:
#     h_falsee = pickle.load(handle)
# with open(truee, 'rb') as handle:
#     h_truee = pickle.load(handle)
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

# mini = min(len(h_falsee),len(h_falsee256), len(h_truee))

# h_falsee256 = h_falsee256[:mini]
# h_falsee = h_falsee[:mini]
# h_truee = h_truee[:mini]
# h_falsee256 = h_falsee256.T
# h_falsee = h_falsee.T
# h_truee = h_truee.T

groups = {"-DRQN": 'blue', "-DQN": 'green',"-DRQN+": 'red',"-Random": 'yellow'}

alel = ['SCORE', 'MAX SCORE', 'AVG SCORE', 'AVG MAX SCORE']
def printPos0(unpackeds,files,lvl):
    plt.clf()
    texts = []
    from adjustText import adjust_text
    for idx, x in enumerate(unpackeds):
        y1 = x[lvl]
        col = 'red'
        print(files[0])
        for tipe in groups:
            if tipe in files[idx]:
                
                col = groups[tipe]
        
        
        plt.plot(y1,color=col, alpha=0.5)
        maxSofar = 0
        for xy in zip(x[4],y1):
            if xy[1] > maxSofar:
                maxSofar = xy[1]
                # texts.append(plt.text(xy[0], xy[1], xy[1]))

    plt.title('Assault - Atari')
    plt.ylabel(alel[lvl])
    plt.xlabel('episode')
    #plt.legend(files, loc='lower right')#'max_step', 'AVG', 'AVG MAX'

    #print(files)
    plt.legend(files, bbox_to_anchor=(1.04,1), borderaxespad=0)
    #plt.subplots_adjust(right=0.7)
    # for var in (y1, y2):
    #     plt.annotate('%0.2f' % var.max(), xy=(1, var.max()), xytext=(4, 0), 
    #              xycoords=('axes fraction', 'data'), textcoords='offset points')

    
    # adjust_text(texts, autoalign='y', expand_objects=(0.1, 1),
    #         only_move={'points':'', 'text':'y', 'objects':'y'}, force_text=0.75, force_objects=0.1,
    #         arrowprops=dict(arrowstyle="simple, head_width=0.25, tail_width=0.05", color='r', lw=0.5, alpha=0.5))
    if lvl == 2:
        plt.gca().set_ylim(top=1100, bottom=500)
    plt.savefig('compa'+str(lvl)+'.png', bbox_inches="tight")

# printPos0(h_falsee,h_truee,h_falsee256,0)
# printPos0(h_falsee,h_truee,h_falsee256,1)
# printPos0(h_falsee,h_truee,h_falsee256,2)
# printPos0(h_falsee,h_truee,h_falsee256,3)

for lvl in range(0,4):
    files = [x.replace(".pickle", "") for x in files]
    printPos0(unpacked, files, lvl)

