# Importing the necessary libaries and the files to use in the value iteration and policy iteration functions

from gridWorld import *

from nextState import nextState
from smallGrid import smallGrid 
from mediumGrid import mediumGrid
from testGrid import testGrid
from costFunction import getCost, getCostBridge
import numpy as np
import copy


actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

def getIndexOf(states, neighbor):
    """
    This function returns the index of all the neighbor states
    of the current states
    """
    neighbor = list(neighbor)
    return states.tolist().index(neighbor)

def getNeighbors(current_state, actions, n, m):
    """
    Description: Returns the neighbors of the current state and the associated actions
                 required to move to those neighbor states
    
    Arguments: Current State: current_state (dtype: tuple)
               Action set: actions (dtype: lsit of tuples)
               No. of columns in the grid-world: n (dtype: int)
               No. of rows in the grid-world: m (dtype: int)
    """
    neighbors = []
    neighbor_acts = []
    for action in actions:
        nb = (current_state[0] + action[0], current_state[1] + action[1])
        if 0 <= nb[0] < n and 0 <= nb[1] < m:
            neighbors.append(nb)
            neighbor_acts.append(action)
        else:
            continue
    neighbors = np.asarray(neighbors)
    return neighbors, neighbor_acts

def getRewards(cost, gridname, states):
    """
    Description: Function returns reward associated with each state
                 given the cost function and the grid type. 
    
    Arguments: Cost function: cost or bridge (dtype: string)
               Gridname: gridname (dtype: string)
               List of states: states (dtype: list)
    """
    stateRewards = []
    for s in states:
        if cost == 'cost':
            r = -1 * getCost(s.tolist(), gridname)
            stateRewards.append(r)
        elif cost == 'bridge':
            r = -1 * getCostBridge(s.tolist(), gridname)
            stateRewards.append(r)
    stateRewards = np.asarray(stateRewards).T
    return stateRewards

def getProb(current_state, current_policy, states, eta, actions, gridname):
    """
    Description: Returns a probability matrix corresponding to each state transition,
                 policy and current state
                 
    Arguments: Current state: current_state (dtype: tuple)
               Policy: current_policy (dtype: tuple)
               List of all states: states (dtype: list)
               Environment uncertainty: eta (dtype: float)
               Action set: actions (dtype: list of tuples)
               Gridname: "gridname" (dtype: string)
    """
    if gridname=='small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    elif gridname=='medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    else:
        raise NameError("Unknown grid")

    probability = np.zeros((n*m))
    if isObstacle(current_state, O):
        return probability
        
    currentIndex = getIndexOf(states, current_state)
    neighbors, neighborsActions = getNeighbors(current_state, actions, n, m)
    neighborIndex = []
    for i in range(len(neighbors)):
        neighborIndex.append(getIndexOf(states,neighbors[i]))

    prob = None
    if current_policy[0] == actions[0][0] and current_policy[1] == actions[0][1]:
        prob = np.array([1 - eta, eta/2, 0, eta/2, 0])
    elif current_policy[0] == actions[1][0] and current_policy[1] == actions[1][1]:
        prob = np.array([eta/2, 1- eta, eta/2, 0, 0])
    elif current_policy[0] == actions[2][0] and current_policy[1] == actions[2][1]:
        prob = np.array([0, eta/2, 1- eta, eta/2, 0])
    elif current_policy[0] == actions[3][0] and current_policy[1] == actions[3][1]:
        prob = np.array([eta/2, 0, eta/2, 1 - eta, 0])
    else:
        prob = np.zeros(5)

    for i, action in enumerate(actions):
        if collisionCheck(current_state, (action[0], action[1]), O):
            prob[4] = prob[4] + prob[i]
            prob[i] = 0

    probability[currentIndex] = prob[4]

    for nb_acts, nb_idx in zip(neighborsActions, neighborIndex):
        if nb_acts[0] == actions[0][0] and nb_acts[1] == actions[0][1]:
            probability[nb_idx] = prob[0]
        elif nb_acts[0]  == actions[1][0] and nb_acts[1] == actions[1][1]:
            probability[nb_idx] = prob[1]
        elif nb_acts[0]  == actions[2][0] and nb_acts[1] == actions[2][1]:
            probability[nb_idx] = prob[2]
        elif nb_acts[0]  == actions[3][0] and nb_acts[1] == actions[3][1]:
            probability[nb_idx] = prob[3]

    return probability

def valueIteration(gamma, cost, eta, gridname):
    """
    Description: Implements value iteration with a discount factor gamma and 
                 the pre-defined cost functions.
      
    Arguments: Discount factor: gamma (dtype: float)
               Cost function: cost (dtype: string)
               Environment Uncertainty: eta (dtype: float)
               Gridname: gridname (dtype: string)
               
    Output: values: Numpy array of (n,m) dimensions
            policy: Numpy array of (n,m) dimensions
    """
    # Use small and medium grid for your code submission
    # cost types: {'cost', 'bridge'}
    if gridname=='small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    elif gridname=='medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    elif gridname=='test':
        n, m, O, START, WINSTATE, DISTANTEXIT, LOSESTATE, LOSESTATES = testGrid()
    else:
        raise NameError("Unknown grid")

    states = []
    for j in range(m):
        for i in range(n):
            states.append((i,j))
    states = np.asarray(states)
    actions = [(1,0), (0,1), (-1,0), (0,-1)]
    actions = np.asarray(actions)
    error = 1e-3 #error value
    values = 10*np.ones(len(states)).T

    policy = []
    for i in range(len(states)):
        pi = (1,0)
        policy.append(pi)
    policy = np.asarray(policy)
    prob = np.zeros((len(states),len(states)))
    iterations = 0
    stateRewards = getRewards(cost, gridname, states)

    while True:
        iterations = iterations + 1

        #policy evaluation
        for k in range(1):
            values_old = values.copy()
            for s in range(len(states)):
                prob[s, :] = getProb(states[s], policy[s], states, eta, actions, gridname)
            values = prob.dot(stateRewards + gamma*values_old)
    
        #policy improvement
        compareActionValues = np.zeros((len(states), len(actions)))
        for action in range(len(actions)):
            actionValues = np.zeros(len(states)).T
            for s in range(len(states)):
                actionValueProbabiilty = getProb(states[s], actions[action], states, eta, actions, gridname)
                actionValues[s] = actionValueProbabiilty.dot(stateRewards + gamma*values)
            compareActionValues[:, action] = actionValues
            
        maximum = np.argmax(compareActionValues,1)
        policyNew = []
        for p_i in range(len(maximum)):
            if maximum[p_i] == 0:
                policyNew.append((1, 0))
            elif maximum[p_i] == 1:
                policyNew.append((0, 1))
            elif maximum[p_i] == 2:
                policyNew.append((-1, 0))
            elif maximum[p_i] == 3:
                policyNew.append((0, -1))
        policyNew = np.asarray(policyNew)
        policy = policyNew.copy()
        
        if not (np.absolute(values - values_old).any() > error):
            break
        else:
            continue
        
    policyCode = []
    for p in policy:
        p = tuple(p.tolist())
        if p == (1, 0):
            policyCode.append(0)
        elif p == (0, 1):
            policyCode.append(1)
        elif p == (-1, 0):
            policyCode.append(2)
        elif p == (0, -1):
            policyCode.append(3)

    policy = np.asarray(policyCode)
    print(len(policy))
    print(n,m)
    policy = np.fliplr(np.flipud(policy.reshape(m, n)).T)
    values = np.fliplr(np.flipud(values.reshape(m, n)).T)
    return values, policy, iterations



def policyIteration(gamma,cost,eta,gridname):
    """
    Description: Implements policy iteration with a discount factor gamma and 
                 the pre-defined cost functions.
      
    Arguments: Discount factor: gamma (dtype: float)
               Cost function: cost (dtype: string)
               Environment Uncertainty: eta (dtype: float)
               Gridname: gridname (dtype: string)
               
    Output: values: Numpy array of (n,m) dimensions
            policy: Numpy array of (n,m) dimensions
    """
    # Use small and medium grid for your code submission
    # cost types: {'cost', 'bridge'}
    if gridname=='small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    elif gridname=='medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    elif gridname=='test':
        n, m, O, START, WINSTATE, DISTANTEXIT, LOSESTATE, LOSESTATES = testGrid()
    else:
        raise NameError("Unknown grid")

    #states of the grid
    states = []
    for j in range(m):
        for i in range(n):
            states.append((i,j))
    states = np.asarray(states)
    actions = [(1,0),(0,1),(-1,0),(0,-1)]
    actions = np.asarray(actions)

    error = 1e-3 #error value

    values = np.zeros(len(states)).T
    actionValues = np.zeros(len(states))

    policy = []
    for i in range(len(states)):
        pi = (1,0)
        policy.append(pi)
    policy = np.asarray(policy)
    prob = np.zeros((len(states),len(states)))
    iterations = 0
    stateRewards = getRewards(cost, gridname, states)

    while True:
        iterations = iterations + 1
        #policy evaluation
        for k in range(10):
            values_old = values.copy()
            for state in range(len(states)):
                prob[state,:] = getProb(states[state], policy[state], states, eta, actions, gridname)
            values = prob.dot(stateRewards+gamma*values_old)
        
        #policy improvement
        compareActionValues = np.zeros((len(states), 4))
        for action in range(len(actions)):
            actionValues = np.zeros(len(states)).T
            for state in range(len(states)):
                actionValueProbability = getProb(states[state], actions[action], states, eta, actions, gridname)
                actionValues[state] = actionValueProbability.dot(stateRewards + gamma*values)
            compareActionValues[:, action] = actionValues
            
        p_argmax = np.argmax(compareActionValues, 1)
        policyNew = []
        for p_i in range(len(p_argmax)):
            if p_argmax[p_i] == 0:
                policyNew.append((1, 0))
            elif p_argmax[p_i] == 1:
                policyNew.append((0, 1))
            elif p_argmax[p_i] == 2:
                policyNew.append((-1, 0))
            elif p_argmax[p_i] == 3:
                policyNew.append((0, -1))
        policyNew = np.asarray(policyNew)
        if np.array_equal(policyNew, policy):
            break
        else:
            policy = policyNew.copy()

    policyCode = []
    for p in policy:
        p = tuple(p.tolist())
        if p == (1, 0):
            policyCode.append(0)
        elif p == (0, 1):
            policyCode.append(1)
        elif p == (-1, 0):
            policyCode.append(2)
        elif p == (0, -1):
            policyCode.append(3)
    policy = np.asarray(policyCode)
    policy = np.fliplr(np.flipud(policy.reshape(m, n)).T)
    values = np.fliplr(np.flipud(values.reshape(m, n)).T)
    return values, policy, iterations

def optimalValues(test):
    """
    Description: Function can be used to testing out different environment
                 conditions and discount factors to visualize the differnt 
                 output paths
    
    Arguments: Testing type: test (dtype: string)
    """
    if question=='a':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='b':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='c':
        gamma=0.6
        eta=0.38
        return gamma, eta
    elif question=='d':
        gamma=0.03
        eta=0
        return gamma, eta
    elif question=='e':
        gamma=0.2
        eta=0.05
        return gamma, eta
    elif question=='f':
        gamma=0.9
        eta=0.1
        return gamma, eta
    elif question=='g':
        gamma=0.9
        eta=0.4


        return gamma, eta
    else: 
        pass
    return 0
    
def showPath(xI,xG,path,n,m,O):
    gridpath = makePath(xI,xG,path,n,m,O)
    fig, ax = plt.subplots(1, 1) # make a figure + axes
    ax.imshow(gridpath) # Plot it
    ax.invert_yaxis() # Needed so that bottom left is (0,0)
    
    
# Function to actually plot the cost-to-gos
def plotValues(values,xI,xG,n,m,O):
    gridvalues = makeValues(values,xI,xG,n,m,O)
    fig, ax = plt.subplots() # make a figure + axes
    ax.imshow(gridvalues) # Plot it
    ax.invert_yaxis() # Needed so that bottom left is (0,0)


def showValues(n,m,values,O):
    string = '------'
    for i in range(0, n):
        string = string + '-----'
    for j in range(0, m):
        print(string)
        out = '| '
        for i in range(0, n):            
            jind = m-j-1 # Need to reverse index so bottom-left is (0,0)
            if isObstacle((i,jind),O):
                out += 'Obs' + ' | '
            else:
                out += str(values[i,jind]) + ' | '
        print(out)
    print(string)       

def showPolicy(n, m, policy, O):
    uSet2 = ["-->", " ^ ", "<--", " v "]
    showValues(n, m, np.array([[uSet2[a] for a in row] for row in policy]), O)

def get_Path(policy, eta, gridname):
    if gridname=='small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    elif gridname=='medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    elif gridname=='test':
        n, m, O, START, WINSTATE, DISTANTEXIT, LOSESTATE, LOSESTATES = testGrid()
    else:
        raise NameError("Unknown grid")

    states = []
    for j in range(m):
        for i in range(n):
            states.append((i,j))
    states = np.asarray(states)
    
    policy = (np.flipud(np.fliplr(policy).T)).reshape((n*m, 1))
    pol = []
    for x in range(len(policy)):
        if policy[x] == 0:
            pol.append((1,0))
        elif policy[x] == 1:
            pol.append((0,1))
        elif policy[x] == 2:
            pol.append((-1,0))
        elif policy[x] == 3:
            pol.append((0,-1))
    policy = pol.copy()
    
    path = []
    current = START
    path.append(current)
    
    while True:
        idx  = getIndexOf(states, current)
        action = policy[idx]
        newState = nextState(current, action, eta, O)
        path.append(newState)
        current = newState
        if gridname == 'small':
            if current[0] == WINSTATE[0] and current[1] == WINSTATE[1]:
                break
        elif gridname == 'medium':
            if current[0] == DISTANTEXIT[0] and current[1] == DISTANTEXIT[1] or \
                current[0] == CLOSEEXIT[0] and current[1] == CLOSEEXIT[1]:
                break
    return path

if __name__ == '__main__':
    gridname = 'small'
    # Use small and medium grid for your code submission
    # cost types: {'cost', 'bridge'}
    if gridname=='small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    elif gridname=='medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    elif gridname=='test':
        n, m, O, START, WINSTATE, DISTANTEXIT, LOSESTATE, LOSESTATES = testGrid()
    else:
        raise NameError("Unknown grid")

    """
    # Case 1:
    """
    # gridname = 'small'
    # gamma, eta = optimalValues('a')
    # cost = 'cost'
    # values, policy, iterations = policyIteration(gamma, cost, eta, gridname)
    # print('The number of iterations for policyIteration algorithm to converge is: ', iterations)
    """
    #Case 2
    """
    gamma, eta = optimalValues('b')
    cost = 'cost'
    values, policy, iterations = valueIteration(gamma, cost, eta, gridname)
    print('The number of iterations for policyIteration algorithm to converge is: ', iterations)
    """
    # Case 3
    """
    # gamma, eta = optimalValues('c')
    # cost = 'cost'
    # values, policy, iterations = valueIteration(gamma, cost, eta, gridname)
    # print('The number of iterations for policyIteration algorithm to converge is: ', iterations)
    """
    # Case 4
    """
    # gamma, eta = optimalValues('d')
    # cost = 'bridge'
    # values, policy, iterations = valueIteration(gamma, cost, eta, gridname)
    # print('The number of iterations for policyIteration algorithm to converge is: ', iterations)

    # # Sample use of plotValues from gridWorld
    # values = np.zeros((n,m))
    # # Loop through values to just assign some dummy/arbitrary data
    # for i in range(n):
    #     for j in range(m):
    #         if not(isObstacle((i,j),O)):
    #             values[i][j] = (n+2*m-2) - (i + 2*j)
    # # This will print those numeric values as console text
    showValues(n,m,values,O)
    showPolicy(n, m, policy, O)

    # This will plot the actual grid with objects as black and values as
    # shades from green to red in increasing numerical order
    xI = START
    # if gridname == 'small':
    xG =WINSTATE
    # elif gridname == 'medium' and cost == 'cost':
    #     xG = CLOSEEXIT
    # elif gridname ==  'medium' and cost == 'bridge':
    # xG = DISTANTEXIT
    grid = create_binary_grid(n, m, O)
    plotValues(grid*values,xI,xG,n,m,O)

    path = get_Path(policy, eta, gridname)
    print(path)
    showPath(xI,xG,path,n,m,O)
    plt.show()
