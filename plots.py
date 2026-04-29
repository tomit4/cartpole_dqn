import matplotlib.pyplot as plt
import torch


def plot_durations(show_result=False, episode_durations=[]):
    """
    Plots episode durations collected during DQN training.

    The function visualizes training progress by plotting the
    duration of each episode over time. A moving average over
    the previous 100 episodes is also displayed to smooth short-term
    fluctuations and highlight overall performance trends.

    Args:
        show_result (bool, optimal):
            If True, displays the final training result plot.
            Otherwise, updates the plot interactively during training.

        episode_duration (list, optimal):
            List containing episode durations collected during training.

    Returns:
        None
    """
    plt.figure(1, figsize=(6, 4))
    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title("Training Results")
    else:
        plt.clf()
        plt.title("Training...")

    plt.xlabel("Training Episodes")
    plt.ylabel("Duration")

    plt.plot(durations_t.numpy(), label="Episode Duration")

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label="100-Episode Moving Average")

    plt.legend()
    plt.tight_layout()
    plt.pause(0.001)


def plot_epsilon(eps_history):
    """
    Plots epsilon decay over training episodes.

    Shows how exploration (random action probability)
    decreases over time in epsilon-greedy policy.
    """

    plt.figure(2, figsize=(5, 4))
    plt.clf()

    plt.title("Exploration Schedule")
    plt.xlabel("Environment Steps")
    plt.ylabel("ε (Probability of Random Action)")

    plt.plot(eps_history, label="Exploration Rate")

    plt.legend()
    plt.tight_layout()
    plt.pause(0.001)
