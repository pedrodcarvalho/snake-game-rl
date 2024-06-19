import tensorflow as tf
from collections import deque
from tensorflow.keras import layers
import numpy as np
import gym
from gym import spaces
import random
import time


ENV_SIZE = 20


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(12,), dtype=np.float32)
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
            apple_x / ENV_SIZE,
            apple_y / ENV_SIZE,
            head_x / ENV_SIZE,
            head_y / ENV_SIZE,
            (head_x - 1) / ENV_SIZE,
            (head_x + 1) / ENV_SIZE,
            (head_y - 1) / ENV_SIZE,
            (head_y + 1) / ENV_SIZE,
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
        if head_x < 0 or head_x >= ENV_SIZE or head_y < 0 or head_y >= ENV_SIZE or self.snake[0] in self.snake[1:]:
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
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                if (i, j) in self.snake:
                    print('S', end='')
                elif (i, j) == self.apple:
                    print('A', end='')
                else:
                    print('.', end='')
            print('')
        print('')


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def play_trained_model(env, model_path):
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.model = tf.keras.models.load_model(model_path, compile=False)
    agent.model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=agent.learning_rate), loss='mse')
    agent.epsilon = 0.0

    state = env.reset()
    state = np.reshape(state, [1, agent.state_size])
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, agent.state_size])
        state = next_state
        total_reward += reward
        time.sleep(0.1)

    print(f'Total score: {total_reward}')


if __name__ == '__main__':
    env = SnakeEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for step_count in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -ENV_SIZE
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f'episode: {e}/{episodes}, score: {step_count}, e: {agent.epsilon:.2f}')
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # Save model every 50 episodes
        if e % 50 == 0:
            agent.model.save(f'snake_dqn_model_{e}.h5', save_format='h5')

    play_trained_model(env, f'snake_dqn_model_{episodes - 1}.h5')
