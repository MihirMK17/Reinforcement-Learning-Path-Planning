# Reinforcement-Learning - Path Planning 
This repository contains the python implementation of reinforcement learning algorithms 

1. Value Iteration: An iterative algorithm that computes the optimal value function for each state until convergence. At each iteration, it calculates the expected utility of each state considering any possible action and the resulting next state. This method eventually outputs a policy that gives the best action to take for each state.
2. Policy Iteration: This algorithm initializes with a random policy, then iteratively evaluates the policy and improves the policy until an optimal policy is achieved. This method comprises two primary steps - policy evaluation and policy improvement.


## Explanation of the files

1. `gridWorld.py`: Plots a grid world, $O$, of dimensions $n$, $m$, as well as a table in the terminal with the value function at each grid celll (state). The grid plot shows the start and final states and, given a sequence of legal states defining a path, it plots the path in the grid environment.
2. `smallGrid.py`: It is a grid world $O$ of small size
3. `mediumGrid.py`: It is a grid world $O$ of medium size
4. `costFunction.py`: Implements stage cost functions such as "bridgecost" or "cost". These are different state transition costs for the gridworlds, making the path planning algorithms output different paths beacause of varied rewards. 
5. `nextState.py`: Defines a markov chain function to simulate the random state transitions from state $x$ after applying action $u$. The function inputs are the current state $x$, a control action $u$ which can be up = $(0, 1)$, down = $(0, -1)$, left = $(-1, 0)$, or  right = $(1, 0)$, a parameter to encode environment uncertainty $\eta$, and a gridworld $O$. The action dependent transition probabilites are encoded in this function as follows. $Pij(u)$ = 0 for $j$ that are farther from $i$ by one step or action move. When applying $u$ at $i$, the probability of moving into the intended direction $i$ + $u$ is $Pij(u)$ = $1$ - $\eta$, but there is an equal chance $Pij(u)$ = $\eta/2$ of falling into a state with 90 degrees of $i$ + $u$. The chance of moving into the opposite direction is $0$. For example if the policy is up, then the agent moves up with probability $1$-$\eta$, othrewise it moves to either left or right with probability $\eta/2$. In this way, with probability $1$ - $\eta$, `nextState` outputs $j = i +  u$, and with probability $\eta/2$, it ourputs $j = i + v$, whhere v is another action in the set up, down , left, right but different from $-u$. But if one of the moves $v$ which is different from $u$ results in tha obstacle, then that probability gets assigned to a move which makes the agent stay i.e. no action is taken.
7. `valueIteration.py`: Contains implementation of Value Iteration and Policy Iteration algorithms. Outputs a path from the specified intial state and goal state.
8. `testGrid.py`: A sample grid for tesing new grid worlds. 

## Demonstration & Figures: 
In this repository, we've conducted several experiments to better understand the behavior and output of our algorithms. The details of these experiments, along with the figures representing their results, are provided below:
### (a) Policy Iteration on 'smallGrid'
Using discount parameter $gamma$ and the cost function for the 'small' grid, how long do we need to apply policy evaluation and policy improvement steps for convergence with $\gamma$ = 0.9 and $\eta$ = 0.2?
- Path Followed: ![Qa)_output_path](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/fb6bcc47-117e-487d-9cfe-de8b438dd214)
- Value Function: ![Qa)_output_values](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/1a336c5c-5a24-44ae-880c-07e59415f03c)
- Optimum Policy: ![Qa)_output_policy](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/38e0baa7-2a4c-4b2d-b2d9-bd0fee9f827c)
- Number of iteratons for policyIteration algorithm to converge: 4

### (b) Value Iteration on 'smallGrid'
Given the same grid and discount parameter, when do we achieve convergence error $\epsilon$ = 1e−3 with $\gamma$ = 0.9 and $\eta$ = 0.2? How does this compare with the results from (a)?
- Path Followed: ![Qb)_output_path](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/04baaff6-ea84-4074-8513-d6c4db5795a6)
- Value Function: ![Qb)iterations_values3](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/1ec8601b-34b4-42e7-acb3-4fc599fe3a3d)
- Optimum Policy: ![Qb)_output_policy](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/84735b59-d45e-4d84-a082-b526edb569ab)
- Number of iterations for valueIteration algorithm to converge: 308

### (c) Experimenting with `mediumGrid`
Utilizing `mediumGrid` and adjusting the discount factor $\gamma$ and noise $\eta$, can our optimal policy go through cell (2,2)?
#### $\gamma$ = 0.8 and $\eta$ = 0.4
- Path Followed: ![0 8, 0 4, path](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/c7fcaea5-2d7f-4ca0-b638-27ff82ea2da8)
- Value Function: ![0 8, 0 4, values](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/b67ef9fe-5007-4f00-b8aa-66bf248cfe94)
- Optimum Policy: ![0 8, 0 4, policy](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/11a3fe54-ef11-4f43-acb4-1983cd228bec)
  
#### $\gamma$ = 0.9 and $\eta$ = 0.2
- Path Followed: ![0 9, 0 2, path](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/198ad06b-94fe-4a5e-84e8-234dde946e4a)
- Value Function: ![0 9, 0 2, values](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/473591bd-38c7-464e-948f-5360b39ccc17)
- Optimum Policy: ![0 9, 0 2, policy](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/07338701-d896-4763-a987-4e7eebc70317)
  
#### $\gamma$ = 0.74 and $\eta$ = 0.38
- Path Followed: ![0 74, 0 38, path](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/b6c4c469-b251-4638-b7e3-4c67ec26b5ff)
- Value Function: ![0 74, 0 38, values](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/b33bb490-7b10-475d-98f3-b7adeba7d670)
- Optimum Policy: ![0 74, 0 38, policy](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/41c4ff92-c964-42a8-a81e-2d925daac878)
  
#### $\gamma$ = 0.6 and $\eta$ = 0.38
- Path Followed: ![0 6, 0 38, path](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/6eb9fce4-c7db-4ca8-b66a-e05b2481bb97)
- Value Function: ![0 6, 0 38, values](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/f72f0da1-9cec-4e5a-b68a-7d2b27f23baf)
- Optimum Policy: ![0 6, 0 38, policy](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/e3160217-3486-4e75-863a-45d4ab3fb87c)

### (d) Path Planning on 'mediumGrid' with `getCostBridge`
In a special scenario with the `mediumGrid` layout, what parameters for $\gamma$ and $\eta$ give rise to various optimal paths under differing conditions?
#### $\gamma$ = 0.03 $\eta$ = 0
- Path Followed: ![i) path](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/f780cc4c-a5a7-4c44-9cb6-9f43db576fa8)
- Optimum Policy: ![i) policy](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/13c6d85e-7e49-494e-a564-9df3546c3ec8)

#### $\gamma$ = 0.02 $\eta$ = 0.05
- Path Followed: ![ii) path](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/bee645ad-2b5e-482a-b6bf-d8d816086d44)
- Optimum Policy: ![ii) policy](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/05c36215-02eb-43a2-9f57-b3fa941bffb5)

#### $\gamma$ = 0.9 $\eta$ = 0.1
- Path Followed: ![iii) path](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/807b2c42-451a-43eb-ac69-cfda5de6f638)
- Optimum Policy: ![iii) policy](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/d0da9711-755a-43c6-92fd-6eef68223fd7)

#### $\gamma$ = 0.9 $\eta$ = 0.4
- Path Followed: ![iv) path](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/4f1d98c9-b454-4b8d-8f67-173ab3750ae3)
- Optimum Policy: ![iv) policy](https://github.com/MihirMK17/Reinforcement-Learning-Path-Planning/assets/123691876/1334ad37-8855-47ff-a93c-d12dd281ec95)
  
**Important**: Ensure all files are located within the same directory/folder for seamless execution.

## Getting Started

1. Clone this repository.
2. Ensure all Python dependencies are installed.
3. Execute the main script to visualize the grid world and computed path.
4. Adjust the cost functions or grid world definitions to experiment with different paths.
