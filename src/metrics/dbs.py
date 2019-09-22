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

def calcAvgOfX(scores, avgx):
    y1 = []

    for pos in range(0, len(scores), avgx):
        y1.append(np.average(scores[pos:pos+avgx]))

    return y1

def genDiffProg(scores_, avg, posx=1, maxpos=1, padding=0.3):
    scores = calcAvgOfX(scores_, avg)
    y1 = []

    for pos in range(0, len(scores)-1, 1):
        y1.append((scores[pos+1] - scores[pos]))

    #x = [(maxpos*i)+posx for i in range(0, len(y1))]
    return y1


