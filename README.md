# Reinforcement-Learning-Model-Free-One-Step-Temporal-Difference-Tabular-Q-Learning-for-Yahtzee
Reinforcement Learning approach to play Yahtzee.

**Based on an POMDP-Tuple (S, A, r, p)**

- S-set of all possible states of the game
- A-set of all possible actions of the game
- r-reward-function as sole empiric feedback for the agent
- p-not-defined function for state transition / their respective probabilities (due to model-free approach)

**Basic characteristics:**
- Markov Property: The process is memoryless; If the current state 's' should present itself, it doesn't matter how it has been reached.
- POMDP usage: Partially Observable due to the fact, that RAM is limited. The information for the current points achieved in the upper half of the Yahtzee-Sheet is compressed. The Agent can only anticipate how many points are missing until getting the bonus from the upper half of the sheet.

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


## Results

All raw data is in the `results` folder, including plots made with R and the corresponding code.

168.37 points maximum after 30,000,000 games of Yahtzee. $\epsilon$ and $\alpha$ were specifically tuned for this number of episodes. Saturation is visible, but expected, since the exploration phase was ending. An extended training run with correspondingly extended decay schedules for $\epsilon$ and $\alpha$ would likely improve the precision of the approximations, and thus the overall score beyond 168 points. This run was intended as a proof of concept, not a fully converged result.

The concept is empirically supported by the short milestones after 5, 10, and 15 million episodes, where exploration was briefly set to 0. Between these milestone-exploitation phases, the score increases — indicating that exploration in between must have meaningfully improved the agent's knowledge.

Interestingly, complex categories (Yahtzee, Four of a Kind) are not only achieved less often, but also improve much more slowly over training than other categories. This suggests that after 30,000,000 episodes, the agent still has too little experience with the rare state-chains leading to these categories — making their Q-value approximations both less accurate and, likely, somewhat misleading.
