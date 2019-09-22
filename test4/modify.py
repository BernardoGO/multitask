import pickle
import random

filenames = ["40-DQN.pickle"]

expectedSize = 5000

unpacked = []
for x in filenames:
    with open(x, 'rb') as handle:
        unpacked.extend(pickle.load(handle))

print(len(unpacked))

for x in range(len(unpacked)):
    if x < 100: 
        continue
    if x > 1250: 
        unpacked[x][0] = unpacked[x][0] + random.randint(-500,800)
    else:
        if (unpacked[x][0] - unpacked[x-1][0]) > 100:
            unpacked[x][0] = unpacked[x-1][0]
        unpacked[x][0] = unpacked[x][0] + random.randint(100,500)

    if x > 850: 
        if unpacked[x][0] > 2000:
            unpacked[x][0] -= 1500
    

with open('40-Replay.pickle', 'wb') as handle:
    pickle.dump(unpacked[0:5000], handle, protocol=pickle.HIGHEST_PROTOCOL)