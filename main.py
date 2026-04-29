import math
import random
import sys
from itertools import count
from typing import cast

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Discrete

from dqn import DQN
from plots import plot_durations, plot_epsilon
from replay_buffer import ReplayBuffer
from transition import Transition

# Instantiate CartPole environment (MDP: states, actions, rewards)
render_human = False
env = (
    gym.make("CartPole-v1", render_mode="human")
    if render_human
    else gym.make("CartPole-v1")
)
action_space = cast(Discrete, env.action_space)

# Device selection for neural network computation (GPU,CPU)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Hyperparameters for DQN training
BATCH_SIZE = 128
GAMMA = 0.99  # discount factor (Bellman equation)
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005  # target network update rate (Polyak averaging)
LR = 0.0003  # learning rate/gradient step size in Bellman regression

# Setup initial environment
n_actions = action_space.n
state, _ = env.reset()
n_observations = len(state)

# Setup initial policy and target DQNs
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

# Initialize target network (frozen copy of policy network)
target_net.load_state_dict(policy_net.state_dict())

# Instantiate the optimizer and replay buffer
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
replay_buffer = ReplayBuffer(10000)

steps_done = 0

episode_durations = []
eps_history = []


def select_action(state):
    """
    Selects an action using an epsilon-greedy exploration strategy.

    Most of the time, the action with the highest predicted Q-value is
    selected from the policy network. Occasionally, a random action is
    chosen to encourage exploration of the environment.

    Args:
        state (Tensor):
            Current environment state represented as a PyTorch tensor.

    Returns:
        Tensor:
            Tensor containing the selected discrete action.
    """
    global steps_done

    # ε-greedy exploration policy (off-policy exploration)
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    eps_history.append(eps_threshold)

    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # argmax_a Q(s, a): greedy policy improvement step
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # random action for exploration
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )


def optimize_model():
    """
    Performs one optimization step for the Deep Q-Network.

    A minibatch of transitions is sampled from the replay buffer to
    compute target Q-values using the Bellman update equation. The
    predicted Q-values from the policy network are then compared against
    three targets using Huber loss.

    Gradients are computed through backpropagation and applied using the
    optimizer to improve the policy network.

    Returns:
        None
    """
    if len(replay_buffer) < BATCH_SIZE:
        return

    # Sample minibatch from replay buffer (off-policy learning)
    transitions = replay_buffer.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Identify terminal vs non-terminal next states for Bellman backup
    # Terminal states have no future value (Q = 0), so we exclude them from bootstrapping
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    # Extract only non-terminal next states for computing max Q(s', a')
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # Minibatch sampling: organize (s_i, a_i, r_i) into tensors for Bellman update
    # Enables vectorized computation of Q(s_i, a_i) over batch
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s, a) for taken actions (policy evaluation step)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # Bellman Optimality target: max_a' Q(s', a') using target network
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    # TD target from Bellman optimality equation
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # TD error minimization (Huber loss for stability)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping: constrains TD-error gradients to prevent unstable Q-value updates
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def main():
    """
    Runs the main training loop for the CartPole DQN agent.

    For each episode:
        - reset the environment
        - select actions using epsilon-greedy exploration
        - observe rewards and next states
        - store transitions in the replay buffer
        - optimize the policy network
        - update the target network

    Episode durations are recorded and plotted during training.

    Returns:
        None
    """
    # Choose training length depending on compute availability
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50
    for _ in range(num_episodes):
        # Reset environment (start new episode)
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for t in count():
            # Select action using epsilon-greedy policy

            action = select_action(state)

            # Step environment (transition, dynamics)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            # Store transition in replay buffer (off-policy dataset)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

            replay_buffer.enqueue(state, action, next_state, reward)

            state = next_state

            # Q-learning update step (policy evaluation + improvement via max)
            optimize_model()

            # Target network soft update (stabilizing moving target)
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            with torch.no_grad():
                # Soft target network update (Polyak averaging)
                # Implements refinement of "frozen target network"
                # 0. save target network parameters: ϕ′ ← ϕ
                # ϕ′ ← τϕ + (1 − τ) ϕ
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = (
                        TAU * policy_net_state_dict[key]
                        + (1 - TAU) * target_net_state_dict[key]
                    )

            # Update the target network
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                plot_durations(show_result=False, episode_durations=episode_durations)
                plot_epsilon(eps_history)
                break


if __name__ == "__main__":
    plt.ion()
    main()
    print("Complete")
    plt.ioff()
    plt.show()
    sys.exit()
