import random
import numpy as np


class RockPaperScissors:
    def __init__(self):
        self.rounds = 2
        self.current_round = 0
        self.agent_choice = None
        self.opponent_choice = None
        self.rewards = {1: 1, -1: -1, 0: 0}
        self.choices = ['rock', 'paper', 'scissors']
        self.total_reward = 0
        self.agent_space = [0, 1, 2]

    def num_states(self) -> int:
        return self.rounds + 1

    def num_actions(self) -> int:
        return len(self.choices)

    def num_rewards(self) -> int:
        return len(self.rewards)

    def reward(self, i: int) -> float:
        return list(self.rewards.values())[i]

    def state_id(self) -> int:
        return self.current_round

    def reset(self):
        self.current_round = 0
        self.agent_choice = None
        self.opponent_choice = None
        self.total_reward = 0

    def is_game_over(self) -> bool:
        return self.current_round >= self.rounds

    def available_actions(self) -> list:
        return self.choices

    def step(self, action: int):
        if action < 0 or action >= len(self.choices):
            raise ValueError("Invalid action")
        if self.current_round == 0:
            self.agent_choice = self.choices[action]
            self.opponent_choice = random.choice(self.choices)
        else:
            self.opponent_choice = self.agent_choice

        result = self.determine_winner(self.agent_choice, self.opponent_choice)
        self.total_reward += self.rewards[result]

        self.current_round += 1

        return self.state_id(), self.rewards[result], self.is_game_over(), {}

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.environment.num_actions() - 1)
        else:
            return np.argmax(self.q_values[state])

    def determine_winner(self, agent, opponent):
        if agent == opponent:
            return 0
        elif (agent == 'rock' and opponent == 'scissors') or \
                (agent == 'scissors' and opponent == 'paper') or \
                (agent == 'paper' and opponent == 'rock'):
            return 1
        else:
            return -1

    def score(self):
        return self.total_reward
