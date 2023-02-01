# Reinforcement-Learning-
This repository contains the python implementation of reinforcement learning algorithms 

1. Value Iteration
2. Policy Iteration

Explanation of the files

1. `gridWorld.py`: Plots a grid world, $O$, of dimensions $n$, $m$, as well as a table in the terminal with the value function at each grid celll (state). The grid plot shows the start and final states and, given a sequence of legal states defining a path, it plots the path in the grid environment.
2. `smallGrid.py`: It is a grid world $O$ of small size
3. `mediumGrid.py`: It is a grid world $O$ of medium size
4. `costFunction.py`: Implements stage cost functions such as "bridgecost" or "cost". These are different state transition costs for the gridworlds, making the path planning algorithms output different paths beacause of varied rewards. 
5. `nextState.py`: Defines a markov chain function to simulate the random state transitions from state $x$ after applying action $u$. The function inputs are the current state $x$, a control action $u$ which can be up = $(0, 1)$, down = $(0, -1)$, left = $(-1, 0)$, or  right = $(1, 0)$, a parameter to encode environment uncertainty $\eta$, and a gridworld $O$. The action dependent transition probabilites are encoded in this function as follows. $Pij(u)$ = 0 for $j$ that are farther from $i$ by one step or action move. When applying $u$ at $i$, the probability of moving into the intended direction $i$ + $u$ is $Pij(u)$ = $1$ - $\eta$, but there is an equal chance $Pij(u)$ = $\eta/2$ of falling into a state with 90 degrees of $i$ + $u$. The chance of moving into the opposite direction is $0$. For example if the policy is up, then the agent moves up with probability $1$-$eta$, othrewise it moves to either left or right with probability $\eta/2$. In this way, with probability $1$ - $\eta$, `nextState` outputs $j = i +  u$, and with probability $\eta/2$, it ourputs $j = i + v$, whhere v is another action in the set up, down , left, right but different from $-u$. But if one of the moves $v$ which is different from $u$ results in tha obstacle, then that probability gets assigned to a move which makes the agent stay i.e. no action is taken.
7. `valueIteration.py`: Contains implementation of Value Iteration and Policy Iteration algorithms. Outputs a path from the specified intial state and goal state.
8. `testGrid.py`: A sample grid for tesing new grid worlds. 
