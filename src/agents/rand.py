import math
import numpy as np
import random
from config import Config
from solvers.random import RANDOMSolver

class randomagent:


    def __init__(self):
        self.epsilon_decay = 0.001
        self.epsilon_min = 0.01
        self.epsilon_max = 1.0
        #epsilon = 1.0
        self.policynet = RANDOMSolver()
        self.targetnet = RANDOMSolver()
        self.dec = 0
        self.reward_discount_factor = 0.20

    def getEpsilon(self, episode):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * episode)

    def choose_action(self, state, episode, network):
        
        self.dec+=1
        #state_j = np.array([state.reshape(Config._ENV_SPACE)])
        #return (0 if np.random.uniform() < 0.5 else 1) if (np.random.random() <= getEpsilon(episode)) else np.argmax(network.network.predict(state_j))

        return random.randint(0, Config._ACTION_SPACE-1)
        
    def get_reward(self, reward, stepcnt):
        return reward # * math.log10((stepcnt+1*reward_discount_factor))

    def play(self, env, startEpisode = 0):
        max_step = 0
        average_max = 0
        last100 = np.zeros(100)
        last100pos = 0
        history = np.array([])
        for episode in range(startEpisode, Config.n_episodes):
        
            if Config.killer.kill_now:
                print("KILLED")
                break
            state = env.reset()
            state_ = None
            done = False
            stepcnt = 0
            sum_reward = 0
            while not done:
                #env.render()
                action = self.choose_action(state, episode, self.targetnet)
                state_, reward, done, extra = env.step(action)
                if done:  # terminal state

                    state_ = None
                sum_reward += reward#get_reward(reward, stepcnt)
                #if max_step > 500:
                #    env.render()

                self.targetnet.remember(reward, state, state_, action, stepcnt)

                
                state = state_
                stepcnt += 1
                #print(stepcnt)
            self.targetnet.replay()
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
        return history