class Parameters:
    def __init__(self, env_name):
        if env_name == 'LineWorld':
            self.size = (10,)
            self.goal_state = 9
            self.rewards = {9: 10}
        elif env_name == 'GridWorld':
            self.size = (3, 3)
            self.goal_state = 8
            self.rewards = {8: 10, 3: -5}
        elif env_name == 'TwoRoundRockPaperScissors':
            self.size = (1,)
            self.goal_state = None
            self.rewards = {0: 1, 1: -1, 2: 0}
        elif env_name == 'MontyHallLevel1':
            self.size = (3,)
            self.goal_state = None
            self.rewards = {0: 1.0, 1: 0.0}
        elif env_name == 'MontyHallLevel2':
            self.size = (5,)
            self.goal_state = None
            self.rewards = {0: 1.0, 1: 0.0}
        else:
            raise ValueError(f"Unknown environment: {env_name}")