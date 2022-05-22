## Model Free Prediction and Control using Monte Carlo Methods

- Dynamic Programming assumes complete knowledge of the environment (MDP). Generally the full knowledge of the how the environment works is not known
- Monte Carlo (MC) can learn directly from experience collecting while interacting with the environment
- These methods involve estimating the empirical mean of the rewards observed from an encountered state till the end of the episode. Since it is an unbiased estimate it converges to the true value function as *N* -> inf
- These updates can only be applied at the end of the episode and hence this method works only for episodic tasks
- MC methods are unbiased but are high variance since it involves multiple random state transitions, actions and rewards (higher variability)
- MC Policy Evaluation: Given a policy, we want to estimate the value state-value function V(s). We sample episodes of experience (state, action, reward, next state) and estimate V(s) to be the reward received from that state onwards averaged across all experiences. The same can be done for action-value function Q(s,a) too.
- MC Control: Idea is the same as Dynamic Programming. We evaluate the policy and improve it greedily. But always acting greedily with respect to the policy (in on-policy methods) does not ensure we explore all states since we do not know the whole environment.
- We need to explore states and actions too in order to ensure the policy is making the best decision after being trained. We take random actions with a probability of epsilon. This learns the optimal epsilon-greedy policy.
- Off-Policy Learning: How can we learn about the actual optimal (greedy) policy while following an exploratory (epsilon-greedy) policy? We can use importance sampling, which weighs returns by their probability of occurring under the policy we want to learn about.