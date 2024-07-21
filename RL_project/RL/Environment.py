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
        self.initial_state = 0

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


class LineWorld(Environment):
    def __init__(self):
        super().__init__(size=(10,), goal_state=9, rewards={9: 10})


class GridWorld(Environment):
    def __init__(self):
        super().__init__(size=(5, 5), goal_state=24, rewards={24: 10})


class TwoRoundRockPaperScissors(Environment):
    def __init__(self):
        super().__init__(size=(1,), goal_state=None, rewards={0: 1, 1: -1, 2: 0})
        self.opponent_first_round = None
        self.agent_first_round = None

    def reset(self):
        self.opponent_first_round = random.randint(0, 2)
        self.agent_first_round = None
        return 0

    def step(self, state, action):
        if state == 0:
            self.agent_first_round = action
            opponent_action = self.opponent_first_round
            reward = self.get_reward(opponent_action, action)
            return 1, reward, False
        elif state == 1:
            opponent_action = self.agent_first_round
            reward = self.get_reward(opponent_action, action)
            return 2, reward, True

    def get_reward(self, opponent_action, agent_action):
        if opponent_action == agent_action:
            return 0
        elif (opponent_action == 0 and agent_action == 1) or (opponent_action == 1 and agent_action == 2) or (
                opponent_action == 2 and agent_action == 0):
            return 1
        else:
            return -1


class MontyHallLevel1(Environment):
    def __init__(self):
        super().__init__(size=(3,), goal_state=None, rewards={0: 1.0, 1: 0.0})
        self.winning_door = None
        self.agent_first_choice = None
        self.revealed_door = None

    def reset(self):
        self.winning_door = random.randint(0, 2)
        self.agent_first_choice = None
        self.revealed_door = None
        return 0

    def step(self, state, action):
        if state == 0:
            self.agent_first_choice = action
            self.revealed_door = self.get_revealed_door()
            return 1, 0, False
        elif state == 1:
            if action == self.agent_first_choice:
                reward = 1.0 if self.agent_first_choice == self.winning_door else 0.0
            else:
                reward = 1.0 if action == self.winning_door else 0.0
            return 2, reward, True

    def get_revealed_door(self):
        for i in range(3):
            if i != self.agent_first_choice and i != self.winning_door:
                return i


class MontyHallLevel2(Environment):
    def __init__(self):
        super().__init__(size=(5,), goal_state=None, rewards={0: 1.0, 1: 0.0})
        self.winning_door = None
        self.agent_choices = []
        self.revealed_doors = []

    def reset(self):
        self.winning_door = random.randint(0, 4)
        self.agent_choices = []
        self.revealed_doors = []
        return 0

    def step(self, state, action):
        if state < 4:
            self.agent_choices.append(action)
            self.revealed_doors.append(self.get_revealed_door())
            return state + 1, 0, False
        elif state == 4:
            if action in self.agent_choices:
                reward = 1.0 if action == self.winning_door else 0.0
            else:
                reward = 1.0 if action == self.winning_door else 0.0
            return 5, reward, True

    def get_revealed_door(self):
        for i in range(5):
            if i not in self.agent_choices and i != self.winning_door:
                return i
