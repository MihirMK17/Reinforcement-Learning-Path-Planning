
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Spring 2022

@author: parth
"""
def testGrid():
   n, m = 7, 7

   O=[[0,6,0,0],[0,0,0,6],[0,6,6,6],[6,6,0,6],
      [2,2,3,4], [4,4,3,3]]

   START = [1,2]
   DISTANTEXIT = [5,3]
   WINSTATE = [3,3]
   LOSESTATE = [4,2]
   LOSESTATES = [[1,1],[2,1],[3,1],[4,1],[5,1],[4,2]]
   return n, m, O, START, WINSTATE, DISTANTEXIT, LOSESTATE, LOSESTATES


