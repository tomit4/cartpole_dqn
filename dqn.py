import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network used to approximate Q(s, a).

    The network maps environment states to predicted Q-values for
    each possible action. These Q-values estimate the expected
    future reward of taking an action from a given state.
    """

    def __init__(self, n_observations, n_actions):
        """
        Initializes the neural network architecture.

        Args:
            n_observations (int):
                Number of input state features from the environment.

            n_actions (int):
                Number of possible discrete actions available to the agent.
        """
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        """
        Performs forward propagation through the network.

        The input state is passed through fully connected layers
        with ReLU activation functions to produce predicted
        Q-values for each action.

        Args:
            x (Tensor):
                Input state tensor

        Returns:
            Tensor:
                Predicted Q-values for all possible actions.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
