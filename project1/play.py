#!/usr/bin/env python

import _init_paths
import os, sys
import numpy as np
from tictactoe import TicTacToe
from tictactoe import Player


if __name__ == '__main__':
    p = Player()
    p.set_regressor_method('mlp')
    p.set_classifier_method('mlp')
    p.agent.load('data/tictac_multi.txt')
    p.agent.train()
    p.run()
