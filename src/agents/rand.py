import math
import numpy as np
import random
from config import Config
from solvers.sqn import SQNSolver

class regular:


    def __init__(self):
        self.epsilon_decay = 0.001
        self.epsilon_min = 0.01
        self.epsilon_max = 1.0
        #epsilon = 1.0
        self.policynet = SQNSolver()
        self.targetnet = SQNSolver()
        self.dec = 0
        self.reward_discount_factor = 0.20

    def getEpsilon(self, episode):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * episode)

    def choose_action(self, state, episode, network):
        
        self.dec+=1
        state_j = np.array([state.reshape(Config._ENV_SPACE)])
        #return (0 if np.random.uniform() < 0.5 else 1) if (np.random.random() <= getEpsilon(episode)) else np.argmax(network.network.predict(state_j))

        return random.randint(0, Config._ACTION_SPACE-1)
        
    def get_reward(self, reward, stepcnt):
        return reward # * math.log10((stepcnt+1*reward_discount_factor))