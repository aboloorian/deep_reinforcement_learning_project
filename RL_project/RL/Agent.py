import numpy as np
#Agent is the class to manage all agent iteration
#The constructor take 2 parameters , Env object and gamma
#environment:object
#gamma:0<gamma<1

class PolicyAgent:
    def __init__(self, environment, parameters,gamma=1.0):
        self.environment = environment
        self.gamma = gamma
        self.policy = {}
        self.state_values = np.zeros(environment.size[0] * environment.size[1])
        self.initialize_policy()
        self.parameters = parameters

    #Initialization policy by random values of actions
    def initialize_policy(self):
        for state in range(self.environment.size[0] * self.environment.size[1]):
            self.policy[state] = np.random.choice(self.environment.actions)

    # Policy Evaluation
    def policy_evaluation(self, iterations=100):
        for _ in range(iterations):
            new_state_values = np.copy(self.state_values) # create immutable to keep orignal state_values
            for state in range(self.environment.size[0] * self.environment.size[1]):
                if state == self.environment.goal_state:#if reach goal job is done just exit
                    continue
                action = self.policy[state]
                next_state = self.environment.get_next_state(state, action)
                reward = self.environment.get_reward(next_state)
                new_state_values[state] = reward + self.gamma * self.state_values[next_state]
            self.state_values = new_state_values

    # Policy improvement (training)
    def policy_improvement(self):
        policy_stable = True
        for state in range(self.environment.size[0] * self.environment.size[1]):
            if state == self.environment.goal_state:
                continue
            old_action = self.policy[state]
            best_action = None #null
            best_value = float('-inf') #negative infinity
            for action in self.environment.actions:
                next_state = self.environment.get_next_state(state, action)
                reward = self.environment.get_reward(next_state)
                value = reward + self.gamma * self.state_values[next_state]
                #find max value
                if value > best_value:
                    best_value = value
                    best_action = action
            self.policy[state] = best_action
            #ensure action converge and stabilized, if during many iteration we see no change in action so it's stabilized
            if best_action != old_action:
                policy_stable = False
        return policy_stable

    # Policy Iteration to reach stability = evaluation+improvement
    def policy_iteration(self):
        is_policy_stable = False
        while not is_policy_stable:
            self.policy_evaluation()
            is_policy_stable = self.policy_improvement()

    def find_best_path_for_goal(self, start_state):
        path = []
        current_state = start_state
        while current_state != self.environment.goal_state:
            path.append(current_state)
            current_action = self.policy[current_state]
            current_state = self.environment.get_next_state(current_state, current_action)
        path.append(self.environment.goal_state)
        return path
