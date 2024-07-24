import numpy as np


class LineWorld:
    def __init__(self):
        self.grid = np.arange(13)
        self.states = self.grid.size
        self.actions = 2
        self.state = 6
        self.forbidden = 0
        self.game_over = False
        self.rewards = np.zeros(self.states)
        self.rewards[3] = 1

        self.transitions = np.zeros((self.states, self.actions, self.states))
        for s in range(self.states):
            if s > 0:
                self.transitions[s, 0, s - 1] = 0.9
                self.transitions[s, 0, s] = 0.1
            if s < self.states - 1:
                self.transitions[s, 1, s + 1] = 0.9
                self.transitions[s, 1, s] = 0.1

        for a in range(self.actions):
            self.transitions[12, a, 12] = 1

        for s in range(self.states):
            for a in range(self.actions):
                total = np.sum(self.transitions[s, a])
                if total == 0:
                    self.transitions[s, a] = np.zeros(self.states)
                else:
                    self.transitions[s, a] /= total

    def num_states(self) -> int:
        return self.grid.size

    def num_actions(self) -> int:
        return self.actions

    def num_rewards(self) -> int:
        return self.rewards.size

    def reward(self, i: int) -> float:
        return self.rewards[i]

    def p(self, s: int, a: int, s_p: int) -> float:
        return self.transitions[s, a, s_p]

    def state_id(self) -> int:
        return self.state

    def reset(self):
        self.state = 6
        self.forbidden = 0
        self.game_over = False

    def display(self):
        print(f"State: {self.state}")

    def is_forbidden(self, action: int) -> int:
        return self.forbidden

    def is_game_over(self) -> bool:
        return self.game_over

    def available_actions(self) -> np.ndarray:
        return np.array([0, 1])

    def step(self, action: int):
        if self.game_over:
            raise ValueError("Game is over")
        if self.is_forbidden(action):
            raise ValueError("Forbidden action")
        next_state = np.random.choice(np.arange(self.states), p=self.transitions[self.state, action])
        reward = self.rewards[next_state]
        self.state = next_state
        self.game_over = self.state == 3
        return self.state, reward, self.game_over, {}

    def score(self):
        return self.rewards[self.state]

    def is_terminal(self, state):
        return state == 12


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
            transition_prob = self.env.p(s, action, s_prime)
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
            transition_prob = self.env.p(s, action, s_prime)
            reward = self.env.reward(s_prime)
            expected_value += transition_prob * (reward + self.gamma * self.value_function[s_prime])
        return expected_value

env = LineWorld()
policy_iteration = PolicyIteration(env)
policy, value_function = policy_iteration.policy_iteration()
print("Politique optimale : ", policy)
print("Fonction de valeur optimale : ", value_function)