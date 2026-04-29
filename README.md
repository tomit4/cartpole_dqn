# Deep Q-Learning Cartpole DQN

## Introduction

This repository contains the source code for a Study.com Course Assignment,
specifically:

[CS 311 - Artificial Intelligence](https://study.com/academy/course/computer-science-311-artificial-intelligence.html).

This project implements a Deep Q-Network (DQN) agent trained to solve the
CartPole-v1 environment from OpenAI Gymnasium.

The agent learns to balance a pole on a moving cart using reinforcement learning
with experience replay and a target network for training stability.

## Algorithm Overview

The implementation uses:

- Deep Q-Network
- Experience replay buffer
- Target network with soft updates (Polyak averaging)
- Epsilon-greedy exploration strategy
- Huber Loss (Smooth L1 loss)
- Gradient clipping for stability

---

## Setup

Note that a basic understanding of `git`, `python`, `pip`, and the UNIX command
line is required to install and run this project. This project was created on a
Linux platform, but it can be run on any OS with the required dependencies.

**Clone The Repo**

First, you will need to clone the repository, and change into the main project
directory.

```sh
git clone https://github.com/tomit4/cartpole_dqn && cd cartpole_dqn
```

**Setting up the Virtual Environment**

To ensure that you do not incur dependency conflicts with your main OS, it is
highly encouraged that you create and utilize a python virtual environment prior
to installing any dependencies and executing the code.

First, once in the main project directory, instantiate a python virtual
environment:

```sh
python -m venv .venv
```

And now, change into the virtual environment:

```sh
source .venv/bin/activate
```

You should see some sort of visual change in your shell prompt that indicates
you are now utilizing the python virtual environment.

**Updating Pip**

Before installing dependencies, it's a good idea to update `pip`:

```sh
python -m pip install --upgrade pip
```

**Installing dependencies**

Once `pip` has been updated, you can install the dependencies using the provided
`requirements.txt` file:

```sh
python -m pip install -r requirements.txt
```

## Running the program

Once the setup instructions have been completed, you can execute the code by
simply using python to execute `main.py`:

```sh
python main.py
```

This will start training the DQN agent. Two live plots will be displayed:

- Episode duration over time (training performance)
- Epsilon decay over time (exploration rate)

**Hardware Acceleration (GPU/CPU)**

By default, the program uses CPU if no GPU is available. CPU training is slower,
so the number of episodes is reduced (50).

- On NVIDIA systems with
  [CUDA](https://nvidia.custhelp.com/app/answers/detail/a_id/2136/~/how-to-install-cuda)
  installed, GPU acceleration will be used (600 episodes).
- On macOS with Apple Silicon,
  [Metal Performance Shaders (MPS)](https://developer.apple.com/documentation/metal)
  will be used (600 episodes).

## Disclaimer

This project is for educational purposes only and is intended as part of a
reinforcement learning coursework assignment. It demonstrates the implementation
of a Deep Q-Network (DQN) agent and is distributed under the BSD-3-Clause
license.
