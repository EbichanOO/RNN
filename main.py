from RNN import simple_rnn
import numpy as np
from matplotlib import pyplot

if __name__ == '__main__':
    t=0
    timedatas = {}
    logs = np.array([[i]for i in range(30)])
    rnn = simple_rnn()
    while(t!=30): #epok
        x = np.sin((t-1)*30)
        y = rnn.forward(x)

        fast_x = np.sin(t*30)
        rnn.backward(y-fast_x, x)
        logs[t] = y-fast_x

        t+=1

    TT = np.arange(0, 30, 1)

    pyplot.figure(figsize=(12, 6))
    pyplot.plot(TT, logs, '--o')
    pyplot.savefig("loop_u_0.0001.png")
    pyplot.clf()
