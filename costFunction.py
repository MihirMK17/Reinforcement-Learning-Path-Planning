from smallGrid import smallGrid 
from mediumGrid import mediumGrid

# from testGrid import testGrid 
# from mediumtestGrid import mediumtestGrid
# this defines the stage cost function for any layout (small or mediumGrid)
# recall the cost = - reward
def getCost(xprime,gridname):
    if gridname=='small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
        if xprime == WINSTATE: # smallGrid win state
            cost = -1
        elif xprime == LOSESTATE: # smallGrid lose state
            cost = 1
        else:
            cost = 0
    elif gridname=='medium':
        flag = False
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
        for i in range(len(LOSESTATES)): # iterate through LOSESTATES
            if xprime == LOSESTATES[i]:
                flag = True
        if xprime == CLOSEEXIT or xprime == DISTANTEXIT: # mediumGrid win states
            cost = -1
        elif flag == True: # mediumGrid lose states
            cost = 1
        else:
            cost = 0
    elif gridname=='test':
        istestGrid = True
    else:
        cost = 0
        raise NameError("Unknown grid")

    return cost

"""
Define a stage cost function for the bridge layout (medium Grid)
"""
def getCostBridge(xprime,gridname):
    # xprime is the next state under an action of the MC
    if gridname=='medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    elif gridname=='test':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumtestGrid()
    else:
        raise NameError("Wrong grid")
    cost = 0
    if xprime == CLOSEEXIT: # Small win state
        cost = -1
    elif xprime == LOSESTATES[5]: # Small lose state
        cost = 1
    elif xprime == DISTANTEXIT: # Big win state
        cost = -10
    else: 
        for i in range(5):
            if xprime == LOSESTATES[i]: # Big lose state
                cost = 10
    return cost


if __name__ == '__main__':
    # Here you can import variables of the Grid worlds
    # from smallGrid import smallGrid 
    # n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    from mediumGrid import mediumGrid
    n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()