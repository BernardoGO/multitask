import random
import gym
import math
import numpy as np
from collections import deque


import pickle
import sys

from solvers.sqn import SQNSolver

import os
import numpy as np
from PIL import Image
from keras import backend as K
import argparse
from util.GracefulKiller import GracefulKiller
from config import Config
from agents import *
from util.output import output

parser = argparse.ArgumentParser()
parser.add_argument("r")
args = parser.parse_args()

run = int(args.r)


### SET CONFIGs

Config.n_episodes = 350

Config.__USE_PRIOR_KNOWLEDGE__ = False#bool(int(args.k))
Config.__TRAIN_LAST_LAYER__ = True#bool(int(args.t))
Config.__COPY_LAST_WEIGHTS__ = False#bool(int(args.c))
Config.__SIZE__ = 128#int(args.s)

Config.__ENV__ = 'AirRaid-v0'
env = gym.make(Config.__ENV__)
Config._ACTION_SPACE = env.action_space.n
Config.num_context = 2
Config._ENV_SPACE = [250, 160, 3]

#Config.__USE_PRIOR_KNOWLEDGE__ = bool(run % 2)
#config.__COMMENT__ = "{}-Size{}-Lw{}-Tl{}-Uk{}".format(__ENV__, __SIZE__, __COPY_LAST_WEIGHTS__, __TRAIN_LAST_LAYER__, __USE_PRIOR_KNOWLEDGE__)

#__ENV__ + "-" + __SIZE__ + "-" + __COPY_LAST_WEIGHTS__ + "-" + __TRAIN_LAST_LAYER__ + "-" + __USE_PRIOR_KNOWLEDGE__









    
    



#policynet.updateFrom(targetnet)

#

#action = 0 if rnd >

####env = gym.make('Atlantis-ram-v0')


Config.killer = GracefulKiller()


#for run in range(3,30):

print(["RUN", run])









#agent = regular.regularagent()
agent = aDrqn.drqnagent()
history = agent.play(env)


print(history)
output.printPos(history, history[-1][-1], str(run))




agent.targetnet.network.save(str(run)+'-fourth.h5')