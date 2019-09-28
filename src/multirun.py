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

parser.add_argument('-r','--run')
parser.add_argument('-c','--contextRun')
parser.add_argument('-e','--episodes')
parser.add_argument('-m','--model')
parser.add_argument('-o','--contextLoad')
parser.add_argument('-f','--folder')

contextsAtari = ['Assault-v0', 'AirRaid-v0']

Config.num_context = 2
args = parser.parse_args()
run = int(args.run)
contextRun = np.eye(Config.num_context)[int(args.contextRun)]
episodes = int(args.episodes)

modelToLoad = None if args.model is None else int(args.model)
contextToLoad = None if args.contextLoad is None else int(args.contextLoad)
Config.__FOLDER__ = args.folder



### SET CONFIGs

Config.n_episodes = episodes

Config.__USE_PRIOR_KNOWLEDGE__ = False if args.model is None else True
Config.__TRAIN_LAST_LAYER__ = True#bool(int(args.t))
Config.__COPY_LAST_WEIGHTS__ = False#bool(int(args.c))
Config.__SIZE__ = 128#int(args.s)
Config.contex =  contextRun

Config.__ENV__ = contextsAtari[int(args.contextRun)]
env = gym.make(Config.__ENV__)
Config._ACTION_SPACE = env.action_space.n

Config._ENV_SPACE = [250, 160, 3]


Config.killer = GracefulKiller()


#for run in range(3,30):

print(["RUN", run])



agent = smallDrqn.smalldrqnagent()



if modelToLoad is not None:
    agent.targetnet.modelToLoad = Config.__FOLDER__+str(modelToLoad)+".h5"
if contextToLoad is not None:
    agent.targetnet.contextToLoad = Config.__FOLDER__+str(contextToLoad)+".h5"

startEpisodes = 0
if contextToLoad is not None:
    agent.loadAll(str(contextToLoad)+".h5")
    startEpisodes = agent.history[-1][-1]
    Config.n_episodes += startEpisodes


Config.n_episodes =int(Config.n_episodes )
agent.targetnet.initialize()
history = agent.play(env, startEpisode = int(startEpisodes))


print(history)
output.printPos(history, history[-1][-1], str(run))


agent.saveAll(str(run)+".h5")

agent.targetnet.network.save(Config.__FOLDER__+str(run)+'.h5')