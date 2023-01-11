# Function Approximation

- The previous examples have considered Table-Lookup cases. These methods do not scale well especially when our action and/or state space is very large. Also if our action/state space is continuous
- Function approximation is a more memory and data efficient way to scale to large/continuous action/state spaces as they can generalize to unseen states and do not have to explicitly compute value functions for each individual state/action.
- We can treat the RL problem as a supervised problem with the Monte Carlo or TD target as a label, and train our function approximator to fit to these labels using an optimization algorithm such as stochastic gradient descent.
- In the case of TD learning, the target depends on the function approximator but we simply ignore it's gradient. This is why these methods are also known as semi-gradient methods.
- Challenge: We have non-stationary (policy changes) and non-iid (strongly correlated in time) data
- For Control, very few convergence guarantees exist. For non-linear approximators there are basically no gurantees at all and in some cases it might even diverge (non linear approximators in TD Learning).
- Experience Replay: Store experience as a dataset, randomize and repeatedly apply SGD. This helps de-correlate the data and works with off-policy methods (DQN)
- In order to stabilize non-linear approximators we can use fixed targets, otherwise there would be alot of variance in the target if we update the targets as well as the Q function at the same time.