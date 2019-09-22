import pickle


filenames = ["28-DQN.pickle", "38-DQN.pickle"]

expectedSize = 5000

unpacked = []
for x in filenames:
    with open(x, 'rb') as handle:
        unpacked.extend(pickle.load(handle))

print(len(unpacked))
with open('40-DQN.pickle', 'wb') as handle:
    pickle.dump(unpacked[0:5000], handle, protocol=pickle.HIGHEST_PROTOCOL)