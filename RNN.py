import numpy as np
import _pickle as pickle

class simple_rnn:

    def __init__(self, file_name, input_dim):
        '''
        try:
            with open(file_name, 'rb') as f:
                params = pickle.load(f)
            for key, val in params
        except:
        '''

        self.params['W1'] = np.random.randn(input_dim)
        self.params['b1'] = np.zeros(input_dim)

        first_out_size = 30

        self.params['W2'] = np.random.randn(first_out_size)
        self.params['b2'] = np.zeros(30)

        self.params['exW'] = np.random.randn(first_out_size)
        self.params['exb'] = np.zeros(30)

        self.time = 0

    def foward(self, x):
        self.col = np.sum(self.NNs(x, 'W1', 'b1'))

        if self.time == 0:
            self.undercol = 0
        else:
            self.col += np.sum(self.undercol * self.params['exW'] -self.params['exb'])

        self.col = self.sigmoid(self.col)
        self.undercol = self.col  # before out

        self.col = np.sum(self.NNs(self.col, 'W2', 'b2'))
        return self.softmax(self.col)

    def backward(self, d):
        self.params['W2'] -= 0.01 *

    def softmax(self, x):
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def NNs(self, x, keys, key):
        x = x * self.params[keys] - self.params[key]
        return x

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))