class LineWorld:
    def __init__(self):
        self.size = (1, 5)
        self.goal_state = 4
        self.rewards = {4: 10}
        self.actions = ['right', 'left']
        self.current_state = 0

    def get_next_state(self, state, action):
        if action == 'right':
            next_state = min(state + 1, self.size[1] - 1)
        elif action == 'left':
            next_state = max(state - 1, 0)
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
        self.current_state = next_state
        return self.current_state, self.get_reward(self.current_state)

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
