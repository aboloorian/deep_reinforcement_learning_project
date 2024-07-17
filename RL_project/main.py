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
param = Parameters((3, 3),8,{8: 10,3:-5},1.0)
# Initialize the Environment
environment = Environment(param.size, param.goal_state, param.rewards)

# # Initialize the Agent for Policy Iteration
# agent = PolicyAgent(environment,param)

# # Perform policy iteration
# agent.policy_iteration()

# # Print the state values and policy
# print("State Values:")
# print(agent.state_values.reshape(param.size))
# print("\nPolicy:")
# for row in range(param.size[0]):
#     for col in range(param.size[1]):
#         state = row * param.size[1] + col
#         if state == param.goal_state:
#             print(" G ", end=" ")
#         else:
#             print(agent.policy[state], end=" ")
#     print()

#  # Find and print the best path from a starting state to the goal state
# start_state = 0
# best_path = agent.find_best_path_for_goal(start_state)
# print("\nBest Path from state 0 to goal:")
# print(best_path)

# Initialize the Agent for Monte Carlo Prediction
agent = MonteCarloAgent(environment, gamma=0.9)  # Create an instance of MonteCarloAgent

# Perform policy iteration
agent.policy_iteration()

# Print the state values and policy
print("State Values:")
print(agent.state_values.reshape(param.size))
print("\nPolicy:")
for row in range(param.size[0]):
    for col in range(param.size[1]):
        state = row * param.size[1] + col
        if state == param.goal_state:
            print(" G ", end=" ")
        else:
            print(agent.policy[state], end=" ")
    print()

# Find and print the best path from a starting state to the goal state
start_state = 0
best_path = agent.find_best_path_for_goal(start_state)
print("\nBest Path from state 0 to goal:")
print(best_path)
