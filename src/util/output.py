
import matplotlib.pyplot as plt
import pickle
from config import Config

class output:
    def printPos(hist,name, tname):
        plt.clf()
        with open(str(tname)+'-'+Config.__COMMENT__()+'.pickle', 'wb') as handle:
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