import random
import gym
import math
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

import pickle
import sys
print("TETE")
from solvers.sqn import SQNSolver
print("TETE2")
import os
import numpy as np
from PIL import Image
from keras import backend as K
import argparse
from util.GracefulKiller import GracefulKiller
from config import Config
from agents.regular import regular



parser = argparse.ArgumentParser()
parser.add_argument("r")
args = parser.parse_args()


run = int(args.r)

###################### QUI 23-28       - CIFAR10



# import argparse

# parser = argparse.ArgumentParser()


# parser.add_argument("k")
# parser.add_argument("t")
# parser.add_argument("c")
# parser.add_argument("s")
# args = parser.parse_args()

#if args.a ==


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
Config._ENV_SPACE = (250, 160, 3,)
print("PASSSOu")
Config.__USE_PRIOR_KNOWLEDGE__ = bool(run % 2)
#config.__COMMENT__ = "{}-Size{}-Lw{}-Tl{}-Uk{}".format(__ENV__, __SIZE__, __COPY_LAST_WEIGHTS__, __TRAIN_LAST_LAYER__, __USE_PRIOR_KNOWLEDGE__)

#__ENV__ + "-" + __SIZE__ + "-" + __COPY_LAST_WEIGHTS__ + "-" + __TRAIN_LAST_LAYER__ + "-" + __USE_PRIOR_KNOWLEDGE__
















def printPos(hist,name, tname):
    plt.clf()
    with open(str(tname)+'-using-'+str(Config.__USE_PRIOR_KNOWLEDGE__)+'-'+Config.__COMMENT__+'.pickle', 'wb') as handle:
        pickle.dump(hist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    hist_ = hist.T
    plt.plot(hist_[1])
    plt.plot(hist_[2])
    plt.plot(hist_[3])
    plt.title('model accuracy')
    plt.ylabel('average')
    plt.xlabel('episode')
    plt.legend(['max_step', 'AVG', 'AVG MAX'], loc='upper left')
    plt.savefig(str(tname)+'accuracy'+str(name)+'.png')

    
    



#policynet.updateFrom(targetnet)

#

#action = 0 if rnd >

####env = gym.make('Atlantis-ram-v0')


killer = GracefulKiller()


#for run in range(3,30):

print(["RUN", run])








max_step = 0
average_max = 0
last100 = np.zeros(100)
last100pos = 0
history = np.array([])

agent = regular()

for episode in range(0, Config.n_episodes):
    #if episode == 0:
    #    episode = 929
    if killer.kill_now:
        print("KILLED")
        break
    state = env.reset()
    state_ = None
    done = False
    stepcnt = 0
    sum_reward = 0
    while not done:
        #env.render()
        action = agent.choose_action(state, episode, agent.targetnet)
        state_, reward, done, extra = env.step(action)
        if done:  # terminal state

            state_ = None
        sum_reward += reward#get_reward(reward, stepcnt)
        #if max_step > 500:
        #    env.render()

        agent.targetnet.remember(reward, state, state_, action, stepcnt)

        
        state = state_
        stepcnt += 1
        #print(stepcnt)
    agent.targetnet.replay()
    if stepcnt > max_step:
        max_step = stepcnt
    last100[last100pos%100] = stepcnt
    last100pos += 1
    if episode <= 100:
        average = np.average(last100[0:last100pos])
    else:
        average = np.average(last100)

    if average > average_max:
        average_max = average

    if len(history) == 0:
        history = [[stepcnt, max_step, average, average_max, episode]]
    else:
        history = np.append(history, [[stepcnt, max_step, average, average_max, episode]], axis=0)    
    print("Episode: {}, Current: {}, MAX: {}, AVERAGE: {}".format(episode, stepcnt,max_step, average))

printPos(history, episode, str(run))




agent.targetnet.network.save(str(run)+'-fourth.h5')