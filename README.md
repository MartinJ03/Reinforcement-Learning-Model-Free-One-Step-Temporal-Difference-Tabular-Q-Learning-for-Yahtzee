# Reinforcement-Learning-Model-Free-One-Step-Temporal-Difference-Tabular-Q-Learning-for-Yahtzee
Reinforcement Learning approach to play Yahtzee.

Based on an POMDP-Tuple (S, A, r, p)
S - set of all possible states of the game
A - set of all possible actions of the game
r - reward-function as sole empiric feedback for the agent
p - not-defined function for state transition / their respective probabilities (due to model-free approach)

**Basic characteristic:**
-Markov Property: The process is memoryless; If the current state 's' should present itself, it doesn't matter how it has been reached.

**Model-Free:** The Agent does not know about possible outcomes or their respective probabilities. 

**One-Step Temporal-Difference:** Given the timestep $t$, the Q-value update rule is:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]
$$

where $\alpha$ is the learning rate and $\gamma$ is the discount factor.

**Tabularity:** Realised via NumPy in float32, taking up 4.4 GB RAM:
```python
def initialize_q_table(states_cardinality, actions_cardinality):
    return np.zeros((states_cardinality, actions_cardinality), dtype=np.float32)
```
