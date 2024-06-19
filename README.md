# Snake Game with Reinforcement Learning

This project demonstrates the use of Reinforcement Learning to train a snake AI using the Deep Q-Network (DQN) algorithm. The project consists of three main scripts: `snake.py`, `snake-env.py`, and `dqn-agent.py`.

## Requirements

- Python 3.x
- Required Python packages:
  - `numpy`
  - `gym`
  - `tensorflow`
  - `curses` (pre-installed with Python on Unix-based systems)

You can install the required packages using the following command (or by running `pip install -r requirements.txt`):

```bash
pip install numpy gym tensorflow
```

## Files

1. **snake.py**: This script implements a classic Snake game using the `curses` library. The game logic includes movement, collision detection, and score tracking.

2. **snake-env.py**: This script defines a custom OpenAI Gym environment for the Snake game, which is used to train the AI agent. It includes state representation, action handling, and reward calculations.

3. **dqn-agent.py**: This script contains the DQN agent implementation. It includes methods for building the neural network, training the model, and making decisions based on the trained model.

4. **run.py**: This script orchestrates the training of the DQN agent using the custom Snake environment. It includes code for training the agent and playing the game with the trained model.

## Usage

### Playing the Snake Game

To play the classic Snake game manually, run:

```bash
python3 snake.py
```

### Training the DQN Agent

To train the DQN agent using the custom Snake environment, run:

```bash
python3 run.py
```

This will train the agent for 1000 episodes and save the model every 50 episodes.

### Running a Trained Model

To play the game using a pre-trained DQN model, ensure the model file is saved in the same directory and run:

```bash
python3 run.py
```

## Example

1. Play the classic Snake game:

```bash
python3 snake.py
```

2. Train the DQN agent:

```bash
python3 run.py
```

The training process will print the episode number, score, and epsilon value after each episode.

3. After training, the agent will play the game using the trained model, and you will see the snake moving automatically based on the learned policy.

## Notes

- Ensure `curses` is available on your system for running `snake.py`.
- The training process can be time-consuming depending on your system's capabilities.
- The environment size and other parameters can be adjusted in the `SnakeEnv` class within `snake-env.py`.

## License

This project is open-source and available under the [MIT License](./LICENSE).
