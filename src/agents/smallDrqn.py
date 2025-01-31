import math
import numpy as np
import random
from config import Config
from solvers.smallDrqn import SQNSolver
import metrics.dbs
import pickle

class smalldrqnagent:


    def __init__(self):
        self.epsilon_decay = 0.0005
        self.epsilon_min = 0.03
        self.epsilon_max = 1.0
        #epsilon = 1.0
        self.policynet = SQNSolver()
        self.targetnet = SQNSolver(loadWeights=True, autoInitialize=False)
        
        self.dec = 0
        self.reward_discount_factor = 0.20
        self.lstm_last = []
        self.lstm_size = 4
        self.lstm_pos = -1

        self.history = None
        self.dbs1 = []


    def loadAll(self, run):
        self.loadLstmLast(run)
        self.loadHistory(run)
        self.loadMemoryTarget(run)

    def saveAll(self, run):
        self.saveHistory(run)
        self.saveLstmLast(run)
        self.saveMemoryTarget(run)


    def loadMemoryTarget(self, run):
        self.targetnet.loadMemory(run)

    def saveMemoryTarget(self, run):
        self.targetnet.saveMemory(run)

    def saveLstmLast(self, run):
        with open(Config.__FOLDER__ +str(run)+'_LstmLast.pickle', 'wb') as handle:
            pickle.dump(self.lstm_last, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def loadLstmLast(self, run):
        with open(Config.__FOLDER__ +str(run)+'_LstmLast.pickle', 'rb') as handle:
            self.lstm_last = pickle.load(handle)


    def saveHistory(self, run):
        with open(Config.__FOLDER__ +str(run)+'_history.pickle', 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def loadHistory(self, run):
        with open(Config.__FOLDER__ +str(run)+'_history.pickle', 'rb') as handle:
            self.history = pickle.load(handle)
        

    def clearKnowledge(self):
        self.targetnet.fullUpdateFrom(self.policynet)
        self.targetnet.fullUpdateFrom(self.policynet)

    def getEpsilon(self, episode):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * episode)

    def choose_action(self, state, episode, network):
        if self.lstm_pos == -1:
            self.lstm_pos += 1
            for x in range(self.lstm_size):
                self.lstm_last.append(state)
        memr = []
        for x in range(self.lstm_pos - self.lstm_size, self.lstm_pos):
            memr.append(self.lstm_last[x])
        
        self.lstm_pos += 1
        self.lstm_pos %= self.lstm_size
        self.dec+=1
        #state_j = np.array([state.reshape(Config._ENV_SPACE)])
        #return (0 if np.random.uniform() < 0.5 else 1) if (np.random.random() <= getEpsilon(episode)) else np.argmax(network.network.predict(state_j))
        if random.random() < self.getEpsilon(episode):
            return random.randint(0, Config._ACTION_SPACE-1)
        else:
            #print("Se")
            return np.argmax(network.network.predict(np.array([memr]))[1][0:Config._ACTION_SPACE])

    def get_reward(self, reward, stepcnt):
        return reward # * math.log10((stepcnt+1*reward_discount_factor))


    def calculateMetrics(self):
        if len(self.history) > 1:
            y1 = metrics.dbs.genDiffProg([self.history[-1][5],self.history[-2][5]], 1)
            self.dbs1.extend(y1)
            #print(self.dbs1)



    def play(self, env, startEpisode = 0):
        max_step = 0
        max_rw= 0
        average_max = 0
        average_rw_max = 0
        last100 = np.zeros(100)
        last100pos = 0
        rw_last100 = np.zeros(100)
        rw_last100pos = 0
        if self.history is None:
            self.history = np.array([])
        for episode in range(startEpisode, Config.n_episodes):
            #if episode == 3000:
            #    print("CLEARING WEIGHTS")
            #    self.clearKnowledge()
            if Config.killer.kill_now:
                print("KILLED")
                break
            state = env.reset()
            state_ = None
            done = False
            stepcnt = 0
            sum_reward = 0
            last_reward = 0
            while not done:
                #env.render()
                action = self.choose_action(state, episode, self.targetnet)
                state_, reward, done, extra = env.step(action)
                if done:  # terminal state

                    state_ = None
                sum_reward += reward#get_reward(reward, stepcnt)
                #if max_step > 0:
                #    env.render()

                self.targetnet.remember(reward, state, state_, action, stepcnt)
                last_reward = reward
                
                state = state_
                stepcnt += 1
                #print(stepcnt)
            #self.calculateMetrics()
            self.targetnet.replay()
            if stepcnt > max_step:
                max_step = stepcnt

            if sum_reward > max_rw:
                max_rw = sum_reward

            last100[last100pos%100] = stepcnt
            last100pos += 1

            rw_last100[rw_last100pos%100] = sum_reward
            rw_last100pos += 1

            if episode <= 100:
                average = np.average(last100[0:last100pos])
                rw_average = np.average(rw_last100[0:rw_last100pos])
            else:
                average = np.average(last100)
                rw_average = np.average(rw_last100)



            if average > average_max:
                average_max = average
            if rw_average > average_rw_max:
                average_rw_max = rw_average

            if len(self.history) == 0:
                self.history = [[stepcnt, max_step, average, average_max, last_reward, sum_reward, episode]]
            else:
                self.history = np.append(self.history, [[stepcnt, max_step, average, average_max, last_reward, sum_reward, episode]], axis=0)    
            print("Episode: {}, Current_EP: {}, MAX_EP: {}, AVERAGE_EP: {},\n\t Current_RW: {}, Current_SUM_RW: {}, MAX_RW: {}, AVERAGE_RW: {}, ".format(episode, stepcnt,max_step, average, last_reward, sum_reward,max_rw, rw_average))
        return self.history