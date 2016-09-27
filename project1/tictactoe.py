#!/usr/bin/env python

import _init_paths
import os, sys
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix

class LinearRegressor(object):
    """docstring for LinearRegressor."""
    def __init__(self):
        super(LinearRegressor, self).__init__()

    def fit(self, X, y):
        XtX = np.dot(X.transpose(), X)
        Xty = np.dot(X.transpose(), y)
        self.w = np.linalg.solve(XtX, Xty)

    def predict(self, X):
        return np.dot(X, self.w)

class TicTacToe(object):
    """docstring for TicTacToe."""
    def __init__(self):
        super(TicTacToe, self).__init__()

        self.lsvm_classifier = LinearSVC(C=1.0, penalty='l2', dual=False, intercept_scaling=1.0)
        self.knn_classifier  = KNeighborsClassifier(n_neighbors=20)
        self.mlp_classifier  = MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=2000)

        self.knn_regressor = KNeighborsRegressor(n_neighbors=20)
        self.mlp_regressor = MLPRegressor(hidden_layer_sizes=(30, 30), max_iter=2000)
        self.lin_regressor = LinearRegressor()

    def set_classifier_method(self, method):
        method = eval('self.' + method + '_classifier')
        self.classifier_handler = method

    def set_regressor_method(self, method):
        method = eval('self.' + method + '_regressor')
        self.regressor_handler = method

    def corrupt_label(self, y, ratio=0.5):
        nr_all = y.shape[0]
        nr_corrupted = int(ratio * nr_all)

        m = np.amax(y)
        if m == 1:
            y[:nr_corrupted, :] *= -1
        else:
            for i in xrange(nr_corrupted):
                y[i,0] = np.random.randint(m)

        return y

    def load(self, data_path):
        db = np.loadtxt(data_path)
        self.X = db[:, :9]
        self.y = db[:, 9:]

    def train(self):
        self.classifier_handler.fit(self.X, self.y.squeeze())
        self.regressor_handler.fit(self.X, self.y.squeeze())

    def predict(self, X, regress=True):

        if regress:
            y = self.regressor_handler.predict(X.reshape(1,-1))
            return np.argmax(y)
        else:
            y = self.classifier_handler.predict(X.reshape(1,-1))
            return y

    def eval_classifier(self, method, K=10, shuffle=True):
        self.set_classifier_method(method)

        nr_all = self.X.shape[0]
        N = range(nr_all)
        if shuffle:
            np.random.shuffle(N)

        nr_val = nr_all/K

        accu = 0
        cm = 0
        for i in xrange(K):
            val_X = self.X[ N[ i*nr_val : (i+1) * nr_val], : ]
            val_y = self.y[ N[ i*nr_val : (i+1) * nr_val], : ]

            if i == 0:
                train_X = self.X[ N[ nr_val: ], :]
                train_y = self.y[ N[ nr_val: ], :]
            elif i == K-1:
                train_X = self.X[ N[ :(K-1) * nr_val], :]
                train_y = self.y[ N[ :(K-1) * nr_val], :]
            else:
                train_X = np.vstack(( self.X[ N[ : i * nr_val], : ], self.X[ N[ (i+1) * nr_val : ], :] ))
                train_y = np.vstack(( self.y[ N[ : i * nr_val], : ], self.y[ N[ (i+1) * nr_val : ], :] ))

            # corrupt data
            # train_y = self.corrupt_label(train_y)

            # # switch train val
            # tmp_X = np.copy(val_X)
            # val_X = np.copy(train_X)
            # train_X = tmp_X
            #
            # tmp_y = np.copy(val_y)
            # val_y = np.copy(train_y)
            # train_y = tmp_y


            self.classifier_handler.fit(train_X, train_y.squeeze())
            pred_y = self.classifier_handler.predict(val_X)
            accu += self.classifier_handler.score(val_X, val_y.squeeze())
            cm = cm + confusion_matrix(val_y.squeeze(), pred_y).astype(np.float32)

        normalized_cm = cm/np.sum(cm, axis=1)
        return accu/K, normalized_cm

    def eval_regressor(self, method, K=10, shuffle=True):
        self.set_regressor_method(method)

        nr_all = self.X.shape[0]
        N = range(nr_all)
        if shuffle:
            np.random.shuffle(N)

        nr_val = nr_all/K

        accu = 0
        for i in xrange(K):
            val_X = self.X[ N[ i*nr_val : (i+1) * nr_val], : ]
            val_y = self.y[ N[ i*nr_val : (i+1) * nr_val], : ]

            if i == 0:
                train_X = self.X[ N[ nr_val: ], :]
                train_y = self.y[ N[ nr_val: ], :]
            elif i == K-1:
                train_X = self.X[ N[ :(K-1) * nr_val], :]
                train_y = self.y[ N[ :(K-1) * nr_val], :]
            else:
                train_X = np.vstack(( self.X[ N[ : i * nr_val], : ], self.X[ N[ (i+1) * nr_val : ], :] ))
                train_y = np.vstack(( self.y[ N[ : i * nr_val], : ], self.y[ N[ (i+1) * nr_val : ], :] ))

            self.regressor_handler.fit(train_X, train_y.squeeze())
            pred_y = self.regressor_handler.predict(val_X)
            pred_y[pred_y > 0.5] = 1.0
            pred_y[pred_y <=0.5] = 0.0
            tmp = np.ones_like(pred_y)

            accu += 1 - np.sum(np.abs(pred_y - val_y))/np.sum(tmp)

        return accu/K

class Player(object):
    """docstring for Player."""
    def __init__(self):
        super(Player, self).__init__()
        self.agent  = TicTacToe()

    def set_classifier_method(self, method):
        self.agent.set_classifier_method(method)

    def set_regressor_method(self, method):
        self.agent.set_regressor_method(method)

    def is_terminate(self, X):
        x = X.reshape((3,3))

        hs = x.sum(axis=0)
        hmax = np.amax(hs)
        hmin = np.amin(hs)

        if hmax == 3:
            return 1
        elif hmin == -3:
            return -1

        vs = x.sum(axis=1)
        vmax = np.amax(vs)
        vmin = np.amin(vs)

        if vmax == 3:
            return 1
        elif vmin == -3:
            return -1

        ds1 = x[0,0] + x[1,1] + x[2,2]
        if ds1 == 3:
            return 1
        elif ds1 == -3:
            return -1


        ds2 = x[0,2] + x[1,1] + x[2,0]
        if ds2 == 3:
            return 1
        elif ds2 == -3:
            return -1

        i = np.where(X==0)[0]
        if len(i) == 0:
            return 2

        return 0

    def run(self, regress=True):
        print 'Game on'
        X = np.zeros((9), dtype=np.float32)

        while True:
            i = int(raw_input('Please take a step: '))
            if i > 8 or i < 0 or X[i] != 0:
                print 'Please enter a valid position'
                continue
            X[i] = 1
            s = self.is_terminate(X)
            if s != 0:
                break

            y = self.agent.predict(X, regress)
            while X[y] != 0:
                y = self.agent.predict(X, regress)

            X[y] = -1
            s = self.is_terminate(X)
            if s != 0:
                break

            print 'Current state:'
            print X.reshape((3,3))

        print 'Game over'
        print 'Final state:'
        print X.reshape((3,3))

        if s == 1:
            print 'You win!'
        elif s == -1:
            print 'AI win.'
        else:
            print 'Draw.'

if __name__ == '__main__':
    tictac = TicTacToe()

    tictac.load('data/tictac_final.txt')

    print tictac.eval_classifier('lsvm')
    print tictac.eval_classifier('knn')
    print tictac.eval_classifier('mlp')

    tictac.load('data/tictac_single.txt')

    print tictac.eval_classifier('lsvm')
    print tictac.eval_classifier('knn')
    print tictac.eval_classifier('mlp')

    tictac.load('data/tictac_multi.txt')

    print tictac.eval_regressor('lin')
    print tictac.eval_regressor('knn')
    print tictac.eval_regressor('mlp')
