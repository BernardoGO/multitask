import pickle


filenames = ["38-DQN.pickle", "52-DRQNNew.pickle", "66-DRQNNew-LSTM2D.pickle"]

expectedSize = 10000

unpacked = []
#for x in filenames:
#    with open(x, 'rb') as handle:
#        unpacked.extend(pickle.load(handle))


with open(filenames[0], 'rb') as handle:
        xt = pickle.load(handle)
        print(len(xt))
        unpacked.extend(xt[0:])

with open(filenames[1], 'rb') as handle:
        xt = pickle.load(handle)
        print(len(xt))
        unpacked.extend(xt[2:3200]+1500)

with open(filenames[2], 'rb') as handle:
        xt = pickle.load(handle)
        print(len(xt))
        unpacked.extend(xt[2:3200])

print(len(unpacked))
with open('testForgetting2.pickle', 'wb') as handle:
    pickle.dump(unpacked[0:expectedSize], handle, protocol=pickle.HIGHEST_PROTOCOL)