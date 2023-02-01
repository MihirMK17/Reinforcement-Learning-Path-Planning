
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from smallGrid import *


# Makes the maze data with obstacles stored as black RGB values,
# free space stored as white RGB values"
def makeMaze(n,m,O):
    # Initialize to lists of 1. for RGB color index = white
    gridvals = [[ [1. for i in range(3)] for col in range(n)] for row in range(m)] 
    # Iterate through each obstacle
    for l in range(len(O)):
        # Find boundaries of current obstacle
        west, east = [O[l][0], O[l][1]]
        south, north = [O[l][2], O[l][3]]
        # Iterate through each cell of obstacle (clunky, but works)
        for i in range(west,east+1):
            for j in range(south,north+1):
                gridvals[j][i] = [0.,0.,0.] # Change entry to RGB black
    return gridvals


# Function to actually plot the maze
def maze(n,m,O):
    gridvals = makeMaze(n,m,O)
    fig, ax = plt.subplots() # make a figure + axes
    ax.imshow(gridvals) # Plot it
    ax.invert_yaxis() # Needed so that bottom left is (0,0)
    #ax.axis('off')


# Checks for collisions given position x, control u, obstacle list O
def collisionCheck(x,u,O):
    # Check input
    if u != (-1,0) and u != (1,0) and u != (0,-1) and u != (0,1):
        print('collision_check error: Invalid input u!')
        return
    nextx = [x[i] + u[i] for i in range(len(x))]
    for l in range(len(O)):
        # Find boundaries of current obstacle
        west, east = [O[l][0], O[l][1]]
        south, north = [O[l][2], O[l][3]]
        # Check if nextx is contained in obstacle boundaries
        if west <= nextx[0] <= east and south <= nextx[1] <= north:
            return True
    # If we iterate through whole list and don't trigger the "if", then no collisions
    return False

# check if an x is an obstacle
def isObstacle(x,O):
    for l in range(len(O)):
        # Find boundaries of current obstacle
        west, east = [O[l][0], O[l][1]]
        south, north = [O[l][2], O[l][3]]
        # Check if x is contained in obstacle boundaries
        if west <= x[0] <= east and south <= x[1] <= north:
            return True
    # If we iterate through whole list and don't trigger the "if", then no collisions
    return False


# Makes a piece of data with obstacles stored as black RGB values,
# free space stored as white RGB values, and path stored as increasing hue of
# yellow RGB values
def makePath(xI,xG,path,n,m,O):
    # Obtain the grid populated with obstacles and free space RGB values first
    gridpath = makeMaze(n,m,O)
    L = len(path)
    # Iterate through the path to plot as increasing shades of yellow
    for l in range(L-1):
        gridpath[path[l][1]][path[l][0]] = [1.,1.,1-l/(L-1)] # white-->yellow
    gridpath[xI[1]][xI[0]] = [0.,0.,1.] # Initial node (plotted as blue)
    gridpath[xG[1]][xG[0]] = [0.,1.,0.] # Goal node (plotted as green)
    return gridpath


# Constructs path list from initial point and list of actions
def getPathFromActions(xI,actions):
    L = len(actions)
    path = []
    nextx = xI
    for l in range(L):
        u = actions[l]
        if u != (-1,0) and u != (1,0) and u != (0,-1) and u != (0,1):
            print('getPath error: Invalid input u!')
            return
        nextx = [nextx[i] + u[i] for i in range(len(nextx))] # nextx = nextx + u
        path.append(nextx) # Builds the path
    return path
        

def create_binary_grid(n, m, O):
    grid = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if isObstacle((i,j), O) == False:
                grid = 1
    return grid

n,m,O, s, w, l = smallGrid()
grid = create_binary_grid(n,m,O)
print(grid)

# values is n x m sized array with cost-to-go values in free-space, and -1 in
# obstacle space
# returns gridpath, which is an RGB piece of data for plotValues
def makeValues(values,xI,xG,n,m,O):
    minval = np.min(values)
    maxval = np.max(values)
    gridvalues = makeMaze(n,m,O)
    for i in range(m):
        for j in range(n):
            x = (j,i)
            currentval = values[j][i]
            if not(isObstacle(x,O)): # If this is is a valid freespace value
                relcurrentval = (currentval-minval)/(maxval-minval) 
                gridvalues[i][j] = [relcurrentval,1-relcurrentval,0.]
    gridvalues[xI[1]][xI[0]] = [0.,0.,1.] # Initial node (plotted as blue)
    gridvalues[xG[1]][xG[0]] = [0.,1.,0.] # Goal node (plotted as green)
    return gridvalues
