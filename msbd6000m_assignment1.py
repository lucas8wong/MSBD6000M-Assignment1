# -*- coding: utf-8 -*-
"""MSBD6000M Assignment1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eohrUGzW75vZACq_AS5AjSUe4CaCJBvK
"""

"""
This module provides an implementation of Q-learning for financial applications.
It includes classes for the environment, agent, and trainer.
"""

import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import random  # Import random for random number generation

class Environment:
    """
    The Environment class represents the financial environment for the Q-learning agent.

    Attributes:
        T (int): Total number of stages.
        p (float): Probability of obtaining high return from risky asset.
        a_ret (float): High return of risky asset.
        b_ret (float): Low return of risky asset.
        riskless_ret (float): Fixed return of riskless asset.
        W_MAX (int): Maximum wealth.
        W_STEP (int): Discretization step for wealth.
        ACTION_STEP (int): Discretization step for actions.
        alpha (float): Parameter for the utility function.
    """

    def __init__(self, T, p, a_ret, b_ret, riskless_ret, alpha, W_MAX, W_STEP, ACTION_STEP):
        """
        Initialize the environment with the given parameters.

        Args:
            T (int): Total number of stages.
            p (float): Probability of obtaining high return from risky asset.
            a_ret (float): High return of risky asset.
            b_ret (float): Low return of risky asset.
            riskless_ret (float): Fixed return of risky asset.
            alpha (float): Parameter for the utility function.
            W_MAX (int): Maximum wealth.
            W_STEP (int): Discretization step for wealth.
            ACTION_STEP (int): Discretization step for actions.
        """
        self.T = T  # Set total number of stages
        self.p = p  # Set probability of obtaining high return from risky asset
        self.a_ret = a_ret  # Set high return of risky asset
        self.b_ret = b_ret  # Set low return of risky asset
        self.riskless_ret = riskless_ret  # Set fixed return of riskless asset
        self.W_MAX = W_MAX  # Set maximum wealth
        self.W_STEP = W_STEP  # Set discretization step for wealth
        self.ACTION_STEP = ACTION_STEP  # Set discretization step for actions
        self.alpha = alpha  # Set parameter for the utility function

        # Construct the list of discretized wealth levels
        self.all_wealth_levels = list(range(0, W_MAX + 1, W_STEP))
        # Create a mapping from wealth levels to their indices
        self.wealth_to_index = {w: i for i, w in enumerate(self.all_wealth_levels)}
        # Create a dictionary of action candidates for each wealth level
        self.action_candidates = {
            w: list(range(0, w + 1, ACTION_STEP)) for w in self.all_wealth_levels
        }

    def get_next_state(self, w, x):
        """
        Calculate the next state based on the current state and action.

        Args:
            w (int): Current wealth.
            x (int): Investment amount.

        Returns:
            int: Next state (wealth).
        """
        # Determine the return of the risky asset
        ret_risky = self.a_ret if random.random() < self.p else self.b_ret
        # Calculate the next wealth level
        w_float = x * (1.0 + ret_risky) + (w - x) * (1.0 + self.riskless_ret)
        # Discretize the next wealth level
        w_next = int(round(w_float / self.W_STEP)) * self.W_STEP
        # Ensure the next wealth level is within bounds
        w_next = max(0, min(w_next, self.W_MAX))
        return w_next  # Return the next wealth level

    def utility(self, w):
        """
        Calculate the exponential utility of wealth.

        Args:
            w (float): Wealth.

        Returns:
            float: Utility value.
        """
        return -np.exp(-self.alpha * w) / self.alpha  # Calculate and return the utility value

    def get_reward(self, t, w):
        """
        Calculate the reward based on the current time step and wealth.

        Args:
            t (int): Current time step.
            w (int): Current wealth.

        Returns:
            float: Reward value.
        """
        if t == self.T - 1:  # If it is the last time step
            return self.utility(w)  # Return the utility as the reward
        return 0.0  # Otherwise, return 0 as the reward


class Agent:
    """
    The Agent class represents the Q-learning agent.

    Attributes:
        env (Environment): The environment in which the agent operates.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon_start (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_end (float): Final epsilon value for epsilon-greedy policy.
        decay_rate (float): Decay rate for epsilon.
    """

    def __init__(self, env, alpha, gamma, epsilon_start, epsilon_end, decay_rate, INITIAL_WEALTH):
        """
        Initialize the agent with the given parameters.

        Args:
            env (Environment): The environment in which the agent operates.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon_start (float): Initial epsilon value for epsilon-greedy policy.
            epsilon_end (float): Final epsilon value for epsilon-greedy policy.
            decay_rate (float): Decay rate for epsilon.
        """
        self.env = env  # Set the environment
        self.alpha = alpha  # Set the learning rate
        self.gamma = gamma  # Set the discount factor
        self.epsilon_start = epsilon_start  # Set the initial epsilon value
        self.epsilon_end = epsilon_end  # Set the final epsilon value
        self.decay_rate = decay_rate  # Set the decay rate for epsilon
        self.INITIAL_WEALTH = INITIAL_WEALTH # Set the initial wealth

        # Initialize the Q-table
        self.max_act_count = max(len(self.env.action_candidates[w]) for w in self.env.all_wealth_levels)
        self.Q = np.zeros((env.T, len(env.all_wealth_levels), self.max_act_count), dtype=float)
        self.action_index_map = self._build_action_index_map()  # Build the action index map

    def _build_action_index_map(self):
        """
        Build the action index map.

        Returns:
            np.ndarray: Action index map.
        """
        # Initialize an empty array for the action index map
        action_index_map = np.empty((self.env.T, len(self.env.all_wealth_levels)), dtype=object)
        for t in range(self.env.T):  # Iterate over all time steps
            for w_idx, w_val in enumerate(self.env.all_wealth_levels):  # Iterate over all wealth levels
                cand = self.env.action_candidates[w_val]  # Get the action candidates for the current wealth level
                # Create a mapping from action values to their indices
                idx_map = {x_val: i for i, x_val in enumerate(cand)}
                action_index_map[t, w_idx] = idx_map  # Store the mapping in the action index map
        return action_index_map  # Return the action index map

    def compute_epsilon(self, episode):
        """
        Compute the epsilon value for epsilon-greedy, decaying exponentially.
        """
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end)*np.exp(-self.decay_rate*episode)

    def choose_action(self, t, w, eps):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            t (int): Current time step.
            w (int): Current wealth.
            eps (float): Epsilon value for epsilon-greedy policy.

        Returns:
            int: Chosen action (investment amount).
        """
        cand_x = self.env.action_candidates[w]  # Get the action candidates for the current wealth level
        w_idx = self.env.wealth_to_index[w]  # Get the index of the current wealth level
        q_vals_for_w = self.Q[t, w_idx, :len(cand_x)]  # Get the Q-values for the current state

        if random.random() < eps:  # With probability eps, choose a random action (exploration)
            return random.choice(cand_x)  # Return a random action
        else:  # Otherwise, choose the best action (exploitation)
            best_idx = np.argmax(q_vals_for_w)  # Get the index of the best action
            return cand_x[best_idx]  # Return the best action

    def update_q_table(self, t, w, x, reward, w_next):
        """
        Update the Q-table based on the current state, action, reward, and next state.

        Args:
            t (int): Current time step.
            w (int): Current wealth.
            x (int): Chosen action (investment amount).
            reward (float): Reward received.
            w_next (int): Next state (wealth).
        """
        w_idx = self.env.wealth_to_index[w]  # Get the index of the current wealth level
        a_idx = self.action_index_map[t, w_idx][x]  # Get the index of the chosen action

        if t < self.env.T - 1:  # If it is not the last time step
            w_next_idx = self.env.wealth_to_index[w_next]  # Get the index of the next wealth level
            n_actions_next = len(self.env.action_candidates[w_next])  # Get the number of actions for the next state
            max_q_next = np.max(self.Q[t + 1, w_next_idx, :n_actions_next])  # Get the maximum Q-value for the next state
        else:  # If it is the last time step
            max_q_next = 0.0  # Set the maximum Q-value for the next state to 0

        old_q = self.Q[t, w_idx, a_idx]  # Get the old Q-value for the current state-action pair
        # Update the Q-value using the Q-learning update rule
        self.Q[t, w_idx, a_idx] = old_q + self.alpha * (reward + self.gamma * max_q_next - old_q)

    def compute_q_diff(self, Q_old):
        """
        Compute the difference between the old and new Q-tables.

        Args:
            Q_old (np.ndarray): Old Q-table.

        Returns:
            float: Sum of absolute differences between the old and new Q-tables.
        """
        return np.sum(np.abs(Q_old - self.Q))  # Calculate and return the sum of absolute differences


class Trainer:
    """
    The Trainer class is responsible for training the Q-learning agent.

    Attributes:
        agent (Agent): The Q-learning agent.
        num_episodes (int): Number of training episodes.
        errors (list): List to store training errors.
        final_wealths (list): List to store final wealth values.
    """

    def __init__(self, agent, num_episodes):
        """
        Initialize the trainer with the given agent and number of episodes.

        Args:
            agent (Agent): The Q-learning agent.
            num_episodes (int): Number of training episodes.
        """
        self.agent = agent  # Set the agent
        self.num_episodes = num_episodes  # Set the number of training episodes
        self.errors = []  # Initialize the list to store training errors
        self.final_wealths = []  # Initialize the list to store final wealth values

    def train(self):
        """
        Train the Q-learning agent.
        """
        for episode in range(self.num_episodes):  # Iterate over all episodes
            # Calculate the current epsilon value
            epsilon = self.agent.compute_epsilon(episode)

            # Copy the old Q-table to calculate the difference later
            Q_before = self.agent.Q.copy()

            # Initialize the initial state
            t = 0  # Set the initial time step to 0
            w_current = self.agent.INITIAL_WEALTH  # Set the initial wealth

            while t < self.agent.env.T:  # Iterate over all time steps
                # Choose an action using the epsilon-greedy policy
                x_action = self.agent.choose_action(t, w_current, epsilon)

                # Execute the action and get the next state and reward
                w_next = self.agent.env.get_next_state(w_current, x_action)
                reward = self.agent.env.get_reward(t, w_next)

                # Update the Q-table
                self.agent.update_q_table(t, w_current, x_action, reward, w_next)

                # Update the current state
                w_current = w_next
                t += 1  # Increment the time step

            # Record the error and final wealth
            diff_val = self.agent.compute_q_diff(Q_before)  # Calculate the Q-table difference
            self.errors.append(diff_val)  # Append the error to the list
            self.final_wealths.append(w_current)  # Append the final wealth to the list

            # Print the log every 1000 episodes
            if (episode + 1) % 1000 == 0:
                # Calculate average Q-diff over last 100 episodes
                recent_qdiffs = self.errors[-100:] if len(self.errors)>=100 else self.errors
                avg_qdiff_last100 = np.mean(recent_qdiffs)

                # Calculate average final wealth over last 100 episodes
                recent_wealth = self.final_wealths[-100:] if len(self.final_wealths)>=100 else self.final_wealths
                avg_wealth_last100 = np.mean(recent_wealth)

                print(f"Episode {episode+1}/{self.num_episodes} | eps={epsilon:.4f} "
                      f"| Avg Q-diff Last 100={avg_qdiff_last100:.4f} "
                      f"| Avg Wealth Last 100={avg_wealth_last100:.2f}")

    def plot_results(self):
        """
        Plot the training results, including training error and final wealth.
        """
        plt.figure(figsize=(12, 5))  # Create a new figure with a specified size

        # Plot the training error
        plt.subplot(1, 2, 1)  # Create a subplot for the training error
        plt.plot(self.errors, label="Training error (delta Q)")  # Plot the training errors
        window_err = 100  # Set the window size for moving average
        if len(self.errors) > window_err:  # If there are enough errors to calculate the moving average
            # Calculate the moving average of the errors
            smoothed_err = [np.mean(self.errors[max(0, i - window_err + 1):i + 1]) for i in range(len(self.errors))]
            plt.plot(smoothed_err, label=f"Err Moving Avg(window={window_err})", color="red", alpha=0.7)
        plt.xlabel("Episode")
        plt.ylabel("Sum of |delta Q|")
        plt.title("Training Error")
        plt.legend()

        # Plot the final wealth
        plt.subplot(1, 2, 2)
        plt.plot(self.final_wealths, label="Final wealth per episode", alpha=0.6)
        window_w = 500
        if len(self.final_wealths) > window_w:
            smoothed = [np.mean(self.final_wealths[max(0, i - window_w + 1):i + 1]) for i in range(len(self.final_wealths))]
            plt.plot(smoothed, label=f"Wealth MA(window={window_w})", color="red", alpha=0.7)

        plt.xlabel("Episode")
        plt.ylabel("Final Wealth")
        plt.title("Final Wealth over Episodes")
        plt.legend()

        plt.tight_layout()
        plt.show()


# =========== Code below remains the same, but we add new code for multiple scenarios ===========
if __name__ == "__main__":
    """
    In the main section, we define multiple experiment groups (scenarios) to reflect
    different market conditions. Each scenario is run separately, re-initializing
    the environment, agent, and trainer, and then plotting the results.
    """

    # Keep these base constants for discretization and episodes, do not modify this docstring.
    T = 10  # Total number of stages
    alpha_utility = 0.001  # Parameter for the utility function
    INITIAL_WEALTH = 1000  # Initial wealth
    W_MAX = 10000  # Maximum wealth
    W_STEP = 50   # Discretization step for wealth
    ACTION_STEP = 50  # Discretization step for actions

    num_episodes = 30000    # Number of training episodes
    alpha = 0.001           # Learning rate
    gamma = 1.0             # Discount factor

    epsilon_start = 0.2     # Initial epsilon value
    epsilon_end = 0.0001    # Final epsilon value
    decay_rate = 0.005      # Decay rate for epsilon

    # NEW: Define several scenarios for p, a_ret, b_ret, and riskless_ret
    scenarios = [
        {
            "name": "Scenario 1",
            "p": 0.8,
            "a_ret": 0.6,
            "b_ret": -0.3,
            "riskless_ret": 0.02
        },
        {
            "name": "Scenario 2",
            "p": 0.7,
            "a_ret": 0.4,
            "b_ret": -0.2,
            "riskless_ret": 0.01
        },
        {
            "name": "Scenario 3",
            "p": 0.6,
            "a_ret": 0.35,
            "b_ret": -0.05,
            "riskless_ret": 0.015
        },
        {
            "name": "Scenario 4",
            "p": 0.9,
            "a_ret": 0.5,
            "b_ret": -0.2,
            "riskless_ret": 0.03
        }
    ]

    # Loop over scenarios to run separate experiments
    for scenario in scenarios:
        print(f"\n===== Running {scenario['name']} =====")
        p = scenario["p"]
        a_ret = scenario["a_ret"]
        b_ret = scenario["b_ret"]
        riskless_ret = scenario["riskless_ret"]

        # Initialize environment with scenario parameters
        env = Environment(
            T=T,
            p=p,
            a_ret=a_ret,
            b_ret=b_ret,
            riskless_ret=riskless_ret,
            alpha=alpha_utility,
            W_MAX=W_MAX,
            W_STEP=W_STEP,
            ACTION_STEP=ACTION_STEP
        )

        # Initialize agent
        agent = Agent(
            env,
            alpha=alpha,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            decay_rate=decay_rate,
            INITIAL_WEALTH=INITIAL_WEALTH
        )

        # Initialize trainer
        trainer = Trainer(agent, num_episodes=num_episodes)

        # Train for the current scenario
        trainer.train()

        # Print final results label
        print(f"Plotting results for {scenario['name']}...")

        # Plot results for this scenario (shows a new figure each loop)
        trainer.plot_results()