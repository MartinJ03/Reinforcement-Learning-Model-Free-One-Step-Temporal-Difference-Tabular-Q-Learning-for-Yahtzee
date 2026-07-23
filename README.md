# Reinforcement-Learning-Model-Free-One-Step-Temporal-Difference-Tabular-Q-Learning-for-Yahtzee
Reinforcement Learning approach to play Yahtzee.

Based on an MDP-Tuple (S, A, r, p)
S - set of all possible states of the game
A - set of all possible actions of the game
r - reward-function as sole empiric feedback for the agent
p - not-defined function for state transition / their respective probabilities (due to model-free approach)

Basic characteristics:
-Markov Property: The process is memoryless; If the current state 's' should present itself, it doesn't matter how it has been reached.
-

Model-Free:
The Agent does not know about possible outcomes or their respective probabilities. 

One-Step Temporal-Difference:
Given the timestamp $$ /in $$
