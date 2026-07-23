# Reinforcement-Learning-Model-Free-One-Step-Temporal-Difference-Tabular-Q-Learning-for-Yahtzee
Reinforcement Learning approach to play Yahtzee.

Based on an POMDP-Tuple (S, A, r, p)
S - set of all possible states of the game
A - set of all possible actions of the game
r - reward-function as sole empiric feedback for the agent
p - not-defined function for state transition / their respective probabilities (due to model-free approach)

**Basic characteristics:**
-Markov Property: The process is memoryless; If the current state 's' should present itself, it doesn't matter how it has been reached.
-POMDP usage: Partially Observable due to the fact, that RAM is limited. The information for the current points achieved in the upper half of the Yahtzee-Sheet is compressed. The Agent can only anticipate how many points are missing until getting the bonus from the upper half of the sheet.

**Model-Free:** The Agent does not know about possible outcomes or their respective probabilities. 

**One-Step Temporal-Difference:** Given the timestep $t$, the Q-value update rule is:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$

where $\alpha$ is the learning rate and $\gamma$ is the discount factor.

**Tabularity:** Realised via NumPy in float32, taking up 4.4 GB RAM:
```python
def initialize_q_table(states_cardinality, actions_cardinality):
    return np.zeros((states_cardinality, actions_cardinality), dtype=np.float32)
```

**Off-Policy:** The ε-greedy behavior policy used to generate training data is decoupled from the target policy being learned — the update rule always bootstraps off the greedy `max` action, regardless of which action was actually taken next (see update rule above).
The target policy is closely linked to Bellmann's principle, stating that an optimal solution to a sequential decision problem has to consist of subsequences of optimal subsolutions. Thus, the primary goal of the target policy is to always choose the `max` action. The behavior policy is then the policy trying to maneuver the agent's learning process. 
The learning process should be done in a way to achieve the optimal approximation of every action linked to a certain state, so that optimal sequences of decisions can be made.
