import numpy as np


class GridWorld:
    def __init__(self):
        self.grid = np.array([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]
        ])
        self.state = 0
        self.actions = np.array([0,  # up
                                 1,  # down
                                 2,  # left
                                 3  # right
                                 ])
        self.rewards = np.array([-1, 0, 1])
        self.forbidden = np.array([1])
        self.game_over = False

    def num_states(self) -> int:
        return self.grid.size

    def num_actions(self) -> int:
        return self.actions.size

    def num_rewards(self) -> int:
        return self.rewards.size

    def reward(self, state: int) -> float:
        if state == 24:
            return 1  # terminal state
        elif state < 0 or state >= self.grid.size:
            return -1  # out of bounds
        else:
            return 0  # all other states

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        # r_index is not used, but the method needs to handle it
        return 1 if s_p == self.next_state(s, a) else 0

    def state_id(self) -> int:
        return self.state

    def reset(self):
        self.state = 0
        self.game_over = False

    def display(self):
        print(self.grid)

    def is_forbidden(self, action: int) -> int:
        return action in self.forbidden

    def is_game_over(self) -> bool:
        return self.game_over

    def available_actions(self) -> np.ndarray:
        return self.actions

    def step(self, action: int):
        if self.is_game_over():
            return

        if self.is_forbidden(action):
            return

        next_state = self.next_state(self.state, action)
        reward = self.reward(next_state)
        self.state = next_state
        self.game_over = self.is_terminal(self.state)
        return reward

    def score(self):
        return self.reward(self.state)

    def next_state(self, state, action):
        if action == 0:
            next_state = state - 5
            if next_state < 0:
                return state
            else:
                return next_state
        elif action == 1:
            next_state = state + 5
            if next_state >= self.grid.size:
                return state
            else:
                return next_state
        elif action == 2:
            next_state = state - 1
            if next_state < 0 or next_state % 5 == 4:
                return state
            else:
                return next_state
        elif action == 3:
            next_state = state + 1
            if next_state >= self.grid.size or next_state % 5 == 0:
                return state
            else:
                return next_state

    def is_terminal(self, state):
        return state == 24


class PolicyIteration:
    def __init__(self, env, gamma=0.9, theta=0.001, max_iter=1000):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iter = max_iter
        self.policy = np.zeros(self.env.num_states(), dtype=int)
        self.value_function = np.zeros(self.env.num_states())

    def policy_iteration(self):
        for _ in range(self.max_iter):
            self.value_function = self.policy_evaluation()

            policy_stable = True
            for s in range(self.env.num_states()):
                old_action = self.policy[s]
                self.policy[s] = self.greedy_policy(s)
                if old_action != self.policy[s]:
                    policy_stable = False

            if policy_stable:
                break

        return self.policy, self.value_function

    def policy_evaluation(self):
        value_function = np.zeros(self.env.num_states())
        for _ in range(self.max_iter):
            delta = 0
            for s in range(self.env.num_states()):
                v = value_function[s]
                value_function[s] = self.expected_value(s)
                delta = max(delta, abs(v - value_function[s]))
            if delta < self.theta:
                break
        self.value_function = value_function
        return value_function

    def expected_value(self, s):
        action = self.policy[s]
        expected_value = 0
        for s_prime in range(self.env.num_states()):
            transition_prob = self.env.p(s, action, s_prime, action)
            reward = self.env.reward(s_prime)
            expected_value += transition_prob * (reward + self.gamma * self.value_function[s_prime])
        return expected_value

    def greedy_policy(self, s):
        actions = self.env.available_actions()
        best_action = actions[0]
        best_value = self.expected_value_for_action(s, actions[0])
        for action in actions[1:]:
            value = self.expected_value_for_action(s, action)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action

    def expected_value_for_action(self, s, action):
        expected_value = 0
        for s_prime in range(self.env.num_states()):
            transition_prob = self.env.p(s, action, s_prime, action)
            reward = self.env.reward(s_prime)
            expected_value += transition_prob * (reward + self.gamma * self.value_function[s_prime])
        return expected_value


env = GridWorld()
policy_iteration = PolicyIteration(env)
policy, value_function = policy_iteration.policy_iteration()
print("Politique optimale : ", policy)
print("Fonction de valeur optimale : ", value_function)