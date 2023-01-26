
def smallGrid():
   n, m = 6, 5

   O=[[0,5,0,0],[0,0,0,4],[0,5,4,4],[5,5,0,4],
      [2,2,2,2]]


   START = [1,1]
   WINSTATE = [4,3]
   LOSESTATE = [4,2]
   return n, m, O, START, WINSTATE, LOSESTATE


