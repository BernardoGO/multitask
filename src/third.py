import random
import gym
import math
import numpy as np
from collections import deque
import pickle

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

Config.n_episodes = 1000

Config.__USE_PRIOR_KNOWLEDGE__ = True#bool(int(args.k))
Config.__TRAIN_LAST_LAYER__ = True#bool(int(args.t))
Config.__COPY_LAST_WEIGHTS__ = False#bool(int(args.c))
Config.__SIZE__ = 128#int(args.s)
Config.contex =  [1,0]

Config.__ENV__ = 'AirRaid-v0'#'Assault-v0','AirRaid-v0'
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
agent = smallDrqn.smalldrqnagent()
#agent.targetnet.modelToLoad = "/home/bernardo/Google Drive/Projects/msc/multitask/data_mt/8742-SmallDRQN1Cont.h5"

agent.targetnet.modelToLoad = "/home/bernardo/Google Drive/Projects/msc/multitask/8798-SmallDRQN1Cont.h5"
agent.targetnet.contextToLoad = "/home/bernardo/Google Drive/Projects/msc/multitask/8794-SmallDRQN1Cont.h5"
#"/home/bernardo/Google Drive/Projects/msc/multitask/data_mt/8738-SmallDRQN1_AirRaid.h5"
#agent = rand.randomagent()
#agent.targetnet.network.save('3066-DQN30001.h5')

agent.targetnet.initialize()
history = agent.play(env, startEpisode = 0)


print(history)
output.printPos(history, history[-1][-1], str(run))




agent.targetnet.network.save(str(run)+'-SmallDRQN1Cont.h5')