import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate,concatenate
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.models import load_model
from keras.layers.wrappers import TimeDistributed
import random
import numpy as np
from config import Config



class SQNSolver:
    def __init__(self):
        
        
        self.memory = []
        self.memory_size = 10000
        self.network = None
        self.qnetwork()

        self.lstm_last = []
        self.lstm_size = 4
        self.lstm_pos = -1



    def qnetwork(self, alpha=0.00025):
        # model = Sequential()

        # model.add(Dense(output_dim=128, activation='relu', input_dim=4))
        # model.add(Dense(output_dim=128, activation='relu', input_dim=4))
        # model.add(Dense(output_dim=2, activation='linear'))
        print(Config._ENV_SPACE)
        model_input_shape = tuple([4 ] + Config._ENV_SPACE)
        a = Input(shape=model_input_shape)
        #tuple([timestep] + list(input_shape) + [num_frame])
        if Config.__USE_PRIOR_KNOWLEDGE__:
            model_main = load_model('third.h5')
            
        conv_1 = TimeDistributed(Conv2D(16, kernel_size=(3, 3), activation='relu', name="conv_1"))(a)

        conv_2 = TimeDistributed(Conv2D(8, (3, 3), activation='relu', name="conv_2"))(conv_1)
        max_1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2), name="max_1"))(conv_2)
        drop_1 = TimeDistributed(Dropout(0.25, name="drop_1"))(max_1)
        flatten_0 = TimeDistributed(Flatten(name="flatten_1"))(drop_1)
        flatten_1 = LSTM(24, activation='relu')(flatten_0)
        dense_1 = Dense(24, activation='relu', name="dense_1")(flatten_1)

        dense_1_probB = Dense(24, activation='relu', name="dense_1_probB")(flatten_1)

        concat_2 = concatenate([dense_1, dense_1_probB])

        drop_2 = Dropout(0.5, name="drop_2")(concat_2)

        
        context = Dense(Config.num_context, activation='softmax', name="context")(drop_2)

        dense_2 = Dense(Config._ACTION_SPACE, activation='softmax', name="dense_2")(drop_2)

        #concat_2 = concatenate([context, dense_2])

        #context = Model(inputs=a, outputs=dense_2)
        model = Model(inputs=a, outputs=[context,dense_2])
        

        

        #model = Model(inputs=a, outputs=[context,dense_2])
        #model = Model(inputs=a, outputs=dense_3_act_3)




        opt = RMSprop(lr=alpha)
        model.compile(loss='mse', optimizer=opt)


        if Config.__USE_PRIOR_KNOWLEDGE__:
            model_dict = dict([(layer.name, layer) for layer in model.layers])
            main_dict = dict([(layer.name, layer) for layer in model_main.layers])

            model_dict['conv_1'].set_weights(main_dict['conv_1'].get_weights())
            
            model_dict['conv_2'].set_weights(main_dict['conv_2'].get_weights())
            
            model_dict['max_1'].set_weights(main_dict['max_1'].get_weights())
            
            model_dict['drop_1'].set_weights(main_dict['drop_1'].get_weights())
            model_dict['dense_1'].set_weights(main_dict['dense_1'].get_weights())
        
            

        self.network = model

    def updateFrom(self, value,rate=0.0001):
        weights_list = value.network.get_weights()
        self.network.set_weights(np.multiply(weights_list,rate))



    def remember(self, reward, state, state_, action, step):
        if self.lstm_pos == -1:
            self.lstm_pos += 1
            for x in range(self.lstm_size):
                self.lstm_last.append(state)

        self.lstm_last[self.lstm_pos % self.lstm_size] = state
        memr = None
        memr_ = None
        #for x in range(self.lstm_pos - self.lstm_size, self.lstm_pos):
        #    memr.append(self.lstm_last[x])
            
        for x in range(self.lstm_pos - self.lstm_size, self.lstm_pos):
            if memr is None:
                memr = np.array(self.lstm_last[x]).reshape((1,250,160,3))
            else:
                arr = np.array(self.lstm_last[x]).reshape((1,250,160,3))
                memr = np.vstack((memr, arr))

        #for x in range(self.lstm_pos - self.lstm_size +1, self.lstm_pos):
        #    memr_.append(self.lstm_last[x-1])
        #memr_.append(state_)

        for x in range(self.lstm_pos - self.lstm_size +1, self.lstm_pos):
            if memr_ is None:
                memr_ = np.array(self.lstm_last[x-1]).reshape((1,250,160,3))
            else:
                arr = np.array(self.lstm_last[x-1]).reshape((1,250,160,3))
                memr_ = np.vstack((memr_, arr))
        if state_ is None:
            state_ = np.array(np.zeros( Config._ENV_SPACE))
        memr_ = np.vstack((memr_, np.array(state_).reshape((1,250,160,3))))

        self.lstm_pos += 1
        self.lstm_pos %= self.lstm_size
        self.memory.append([reward, np.array(memr), np.array(memr_), action, step])
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def get_nostate(self):
        no_state = np.array(np.zeros((4, *Config._ENV_SPACE)))
        print("nostate")
        return no_state

    def replay(self):
        GAMMA = 0.99
        batch_size = min(32, len(self.memory))
        con = np.array([Config.contex for x in range(batch_size)])
        
        batch = random.sample(self.memory, batch_size)
        no_state = np.array(np.zeros((4, *Config._ENV_SPACE)))
        states = np.array([ o[1] for o in batch ])

        states_ = []

        for o in batch:
            appn = o[2]
            if o[2] is None:
                appn = no_state
            for x in o[2]:
                if x is None:
                    appn = no_state
                    print("NO STATE")
            #else:
            #    print("STATE")
            states_.append(appn.reshape((4,250,160,3)))
        #(4,250,160,3) 
        states_ = np.array(states_ )
        #states_ = np.array([ (no_state if o[2] is None else o[2]) for o in batch ])



        p = self.network.predict(states)[1]
        print("Replaying")
        p_ = self.network.predict(states_)[1]
        x = np.zeros((batch_size,4, *Config._ENV_SPACE))
        y = np.zeros((batch_size, Config._ACTION_SPACE))

        for idx, single in enumerate(batch):
            reward, state, state_, action, step = single
            t = p[idx]
            if state_ is None:
                t[action] = reward
            else:
                t[action] = reward + GAMMA * np.amax(p_[idx][0:Config._ACTION_SPACE])
            x[idx] = state
            y[idx] = t
        self.network.fit(x, [con, y], batch_size=2, nb_epoch=1, verbose=False)
        # main_dict = dict([(layer.name, layer) for layer in self.network.layers])
        # print(main_dict["dense_2b"].get_weights()[0])