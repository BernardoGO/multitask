import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Concatenate,concatenate
from keras.optimizers import Adam, RMSprop
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.models import load_model
import random
import numpy as np
from config import Config



class RANDOMSolver:
    def __init__(self):
        
        
        self.memory = []
        self.memory_size = 100000
        self.network = None
        self.qnetwork()


    def qnetwork(self, alpha=0.00025):
        # model = Sequential()

        # model.add(Dense(output_dim=128, activation='relu', input_dim=4))
        # model.add(Dense(output_dim=128, activation='relu', input_dim=4))
        # model.add(Dense(output_dim=2, activation='linear'))
        print(Config._ENV_SPACE)
        a = Input(shape=Config._ENV_SPACE)

        if Config.__USE_PRIOR_KNOWLEDGE__:
            model_main = load_model('third.h5')
            
        conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu', name="conv_1")(a)

        conv_2 = Conv2D(24, (3, 3), activation='relu', name="conv_2")(conv_1)
        max_1 = MaxPooling2D(pool_size=(2, 2), name="max_1")(conv_2)
        drop_1 = Dropout(0.25, name="drop_1")(max_1)
        flatten_1 = Flatten(name="flatten_1")(drop_1)
        dense_1 = Dense(24, activation='relu', name="dense_1")(flatten_1)

        dense_1_probB = Dense(64, activation='relu', name="dense_1_probB")(flatten_1)

        concat_2 = concatenate([dense_1, dense_1_probB])

        drop_2 = Dropout(0.5, name="drop_2")(concat_2)


        context = Dense(Config.num_context, activation='softmax', name="context")(dense_1)
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
        #self.network.set_weights(np.multiply(weights_list,rate))

    def remember(self, reward, state, state_, action, step):
        #self.memory.append([reward, state, state_, action, step])
        #if len(self.memory) > self.memory_size:
        #    self.memory.pop(0)
        pass

    def replay(self):
        pass