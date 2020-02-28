import numpy as np
import pickle
import sys
import pkmn
from game_model import *
from x import x

x = np.matrix(x)
# This is very specific to x.py
action_space = ['switch ledian', 'switch swellow', 'move 1', 'move 2', 'move 3', 'move 4']

pkmn.bookkeeper.our_active = 3
pkmn.bookkeeper.opponent_active = 3
record = pkmn.choose_action(x, pkmn.bookkeeper, action_space)
cs0 = []
cs1 = []
cs2 = []
DIFFTHRESH = 0.1

for i in range(OFFSET_HEALTH, N):
    pvev = x[i,0]
    if x[i,0] == 0.0:
        x[i,0] = 1.0
        cs = 0
    elif x[i,0] == 1.0:
        x[i,0] = 0.0
        cs = 1
    else:
        x[i,0] = 3.14
        cs = 2
    newrecord = pkmn.choose_action(x, pkmn.bookkeeper, action_space)
    for j in range(10):
        if POSSIBLE_ACTIONS[j] in action_space:
            if record[j] - newrecord[j] < -1.0 * DIFFTHRESH or record[j] - newrecord[j] >  DIFFTHRESH:
                print(str(cs) + " " + str(i) + " " + str(j) + " " + str(record[j]) + str(newrecord[j]))
                if cs == 0:
                    cs0.append(record[j,0] - newrecord[j][0])
                elif cs == 1:
                    cs1.append(record[j,0] - newrecord[j][0])
                elif cs == 2:
                    cs2.append(record[j,0] - newrecord[j][0])
    x[i,0] = pvev
    assert(np.all(pkmn.choose_action(x, pkmn.bookkeeper, action_space) == record))


