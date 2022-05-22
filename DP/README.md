## Model Based RL: Dynamic Programming

We explore the following concepts:
- **Dynamic Programming**: Helps with solving the Markov Decision Process (MDP) with the assumption that we have a perfect model of the environment. The MDP satisfies the optimal substructure property through the Bellman Equations, as well as the overlapping subproblems since Value functions are stored and reused.
- **Policy Evaluation**: Evaluating a given policy by estimating the state-value function V(s) through iterative application of the Bellman Expectation Backup. This is done via full backups in DP (not very computationally efficient)
- Full Backups use every state and action during it's one step lookahead to evaluate the state-value function V(s). This is only possible if we have a perfect model of the environment since we need to know the state-transition probabilities.
- **Policy Iteration**: This improves the current policy, or fixes it if optimal by acting greedily to an already evaluated state-value function.
- **Value Iteration**: There is no concept of evaluating or improving the policy in this. It is the iterative application of the bellman optimality equation to get the optimal value function from which we extract the optimal policy. It can be thought of as doing a policy evaluation with k=1 steps and improving greedily immediately afterwards (since evaluating a greedy policy is basically taking a max since there would be 1 action with probability, while the rest would be 0)
