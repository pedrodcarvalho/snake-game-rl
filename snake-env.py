import numpy as np
import gym
from gym import spaces
import random


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(24,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.snake = [(5, 5)]
        self.apple = self._generate_apple()
        self.score = 0
        self.done = False
        self.direction = random.choice([0, 1, 2, 3])
        return self._get_state()

    def _generate_apple(self):
        while True:
            apple = (random.randint(0, 9), random.randint(0, 9))
            if apple not in self.snake:
                return apple

    def _get_state(self):
        head_x, head_y = self.snake[0]
        apple_x, apple_y = self.apple
        state = [
            int(self.direction == 0),
            int(self.direction == 1),
            int(self.direction == 2),
            int(self.direction == 3),
            apple_x / 10,
            apple_y / 10,
            head_x / 10,
            head_y / 10,
            (head_x - 1) / 10,
            (head_x + 1) / 10,
            (head_y - 1) / 10,
            (head_y + 1) / 10,
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, self.done, {}

        self._update_direction(action)
        self._move_snake()
        reward = self._check_apple()
        self.done = self._check_collisions()
        return self._get_state(), reward, self.done, {}

    def _update_direction(self, action):
        if action == 0 and self.direction != 1:
            self.direction = 0
        elif action == 1 and self.direction != 0:
            self.direction = 1
        elif action == 2 and self.direction != 3:
            self.direction = 2
        elif action == 3 and self.direction != 2:
            self.direction = 3

    def _move_snake(self):
        head_x, head_y = self.snake[0]
        if self.direction == 0:
            head_x -= 1
        elif self.direction == 1:
            head_x += 1
        elif self.direction == 2:
            head_y -= 1
        elif self.direction == 3:
            head_y += 1
        self.snake.insert(0, (head_x, head_y))

    def _check_collisions(self):
        head_x, head_y = self.snake[0]
        if head_x < 0 or head_x >= 10 or head_y < 0 or head_y >= 10 or self.snake[0] in self.snake[1:]:
            return True
        return False

    def _check_apple(self):
        if self.snake[0] == self.apple:
            self.apple = self._generate_apple()
            self.score += 1
            return 1
        else:
            self.snake.pop()
            return 0

    def render(self, mode='human'):
        for i in range(10):
            for j in range(10):
                if (i, j) in self.snake:
                    print('S', end='')
                elif (i, j) == self.apple:
                    print('A', end='')
                else:
                    print('.', end='')
            print('')
        print('')
