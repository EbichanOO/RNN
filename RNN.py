import numpy as np
import _pickle as pickle

class simple_rnn:

    def __init__(self, input_dim = 1):
        '''
        try:
            with open(file_name, 'rb') as f:
                params = pickle.load(f)
            for key, val in params
        except:
        '''
        self.params = {}

        self.params['W1'] = np.random.randn(input_dim)
        self.params['b1'] = np.zeros(input_dim)

        first_out_size = 30

        self.params['W2'] = np.random.randn(first_out_size)
        self.params['b2'] = np.zeros(30)

        self.params['exW'] = np.random.randn(first_out_size)
        self.params['exb'] = np.zeros(30)

        self.time = 0

    def forward(self, x):
        self.col = np.sum(self.NNs(x, 'W1', 'b1'))

        if self.time == 0:
            self.undercol = 0
            self.unundercol = 0
        else:
            self.unundercol = self.undercol
            self.col += np.sum(self.undercol * self.params['exW'] -self.params['exb'])

        self.col = self.sigmoid(self.col)
        self.undercol = self.col  # before out

        self.col = np.sum(self.NNs(self.col, 'W2', 'b2'))

        out = self.col / 30
        #活性化関数エラー
        return out

    def backward(self, d, x):
        self.params['W2'] -= 0.0001 * np.sum(d * self.undercol)
        self.params['W1'] -= 0.0001 * np.sum(d * x)

        if self.time != 0:
            self.params['exW'] -= 0.0001 * np.sum(d * self.unundercol)

        self.time += 1

    def softmax(self, x): #crazy
        #print(x.shape)
        return np.sum(x)/x.shape

    def NNs(self, x, keys, key):
        x = x * self.params[keys] - self.params[key]
        return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
