#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tor
"""

import numpy as np
from gridWorld import collisionCheck

def nextState(x,u,eta,O):
    if u != (-1,0) and u != (1,0) and u != (0,-1) and u != (0,1):
        print('nextState error: Invalid input u!')
        return
    p = np.random.random_sample()
    if p > eta: # No movement error
        if collisionCheck(x,u,O): # if u is in an obstacle
            return x
        else:
            nextx = [x[i] + u[i] for i in range(len(x))] # nextx = nextx + u
            return nextx
    else: # Movement error, switch up direction
        if 0 <= p <= eta/2:
            if u == (-1,0) or u == (1,0):
                u = (0,1)
            elif u == (0,1) or u == (0,-1):
                u = (1,0)
        elif eta/2 <= p <= eta:
            if u == (-1,0) or u == (1,0):
                u = (0,-1)
            elif u == (0,1) or u == (0,-1):
                u = (-1,0)
        if collisionCheck(x,u,O): # if the new u is in an obstacle
            return x
        else:
            nextx = [x[i] + u[i] for i in range(len(x))] # nextx = nextx + u
            return nextx

