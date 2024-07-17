#Environment is the class to manage all related Env values
#The constructor take size of game board (env) , state of goal and reward distribution
#Size:size of board (m*n)
#goal_state:int numbers representing state
#rewards:map, reward distribution=>example {8: 10}
class Environment:
    def __init__(self, size, goal_state, rewards):
        self.size = size
        self.goal_state = goal_state
        self.rewards = rewards
        self.actions = ['up', 'down', 'left', 'right']
        self.initial_state = 0  # assuming the initial state is 0

    def reset(self):
        return self.initial_state

    def step(self, state, action):
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(next_state)
        done = next_state == self.goal_state
        return next_state, reward, done

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
        return row * self.size[1] + col

    def get_reward(self, state):
        return self.rewards.get(state, -1)
