import numpy as np


class GridWorld:
    def __init__(self):
        self.size = (6, 6)
        self.goal_state = 8
        self.rewards = {8: 10, 3: -5}
        self.actions = ['up', 'down', 'left', 'right']
        self.current_state = 0
        self.action_space = [0, 1, 2, 3]

    def get_next_state(self, state, action):
        row, col = divmod(state, self.size[1])

        if action == 'up':
            row = max(row - 1, 0)
        elif action == 'down':
            row = min(row + 1, self.size[0] - 1)
        elif action == 'left':
            col = max(col - 1, 0)
        elif action == 'right':
            col = min(col + 1, self.size[1] - 1)

        next_state = row * self.size[1] + col
        return next_state

    def get_reward(self, state):
        return self.rewards.get(state, -1)

    def reset(self):
        self.current_state = 0
        return self.current_state

    def is_game_over(self):
        return self.current_state == self.goal_state

    def step(self, action):
        next_state = self.get_next_state(self.current_state, action)
        reward = self.get_reward(next_state)
        done = self.is_game_over()
        self.current_state = next_state
        return next_state, reward, done, {}

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.environment.num_actions())
        else:
            return np.argmax(self.q_values[state])

    def num_states(self):
        return self.size[0] * self.size[1]

    def num_actions(self):
        return len(self.actions)

    def available_actions(self):
        return self.actions

    def state_id(self):
        return self.current_state

    def reward(self, state):
        return self.rewards.get(state, -1)

    def get_state_value(self, state):
        state_values = [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35]
        ]
        return state_values[state // self.size[1]][state % self.size[1]]
