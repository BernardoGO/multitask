import pickle
import random

filenames = ["Partial.pickle"]

expectedSize = 5000

unpacked = []
#for x in filenames:
#    with open(x, 'rb') as handle:
#        unpacked.extend(pickle.load(handle))


with open(filenames[0], 'rb') as handle:
    unpacked.extend(pickle.load(handle)[0:3000])

for pos in range(2000, 2100):
    unpacked[pos][0] -= random.randint(200,800)


print(len(unpacked))
with open('PartialForgetting.pickle', 'wb') as handle:
    pickle.dump(unpacked[0:expectedSize], handle, protocol=pickle.HIGHEST_PROTOCOL)