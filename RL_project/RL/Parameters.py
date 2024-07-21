class Parameters:
    def __init__(self, size, goal_state, rewards,gamma):
        self.size = size
        self.goal_state = goal_state
        self.rewards = rewards
        self.gamma = gamma