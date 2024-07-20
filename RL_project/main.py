# Parameters
import numpy as np

from RL.Agent import PolicyAgent
from RL.Agent import MonteCarloAgent
from RL.Environment import Environment
from RL.Parameters import Parameters
# Prepare input data
#Example
# +---+---+---+
# | 0 | 0 | 0|
# +---+---+---+
# | -5 | 0| 0 |
# +---+---+---+
# |S  | 0 | 10 |
# +---+---+---+
#
# Exemple d'utilisation
param = Parameters((3, 3), 8, {8: 10, 3: -5}, 1.0)
environment = Environment(param.size, param.goal_state, param.rewards)

# Utilisation de l'agent de politique
policy_agent = PolicyAgent(environment, param)
policy_agent.policy_iteration()

print("Valeurs des états (Policy Agent):")
print(policy_agent.state_values.reshape(param.size))
print("\nPolitique (Policy Agent):")
for row in range(param.size[0]):
    for col in range(param.size[1]):
        state = row * param.size[1] + col
        if state == param.goal_state:
            print(" G ", end=" ")
        else:
            print(policy_agent.policy[state], end=" ")
    print()

best_path = policy_agent.find_best_path_for_goal(start_state)
print("\nMeilleur chemin (Policy Agent) de l'état 0 à l'objectif:")
print(best_path)
