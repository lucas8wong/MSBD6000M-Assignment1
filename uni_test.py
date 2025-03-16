import unittest
import numpy as np
from asset_allocation_qlearning import Environment, Agent, Trainer  


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        """ Initialize the environment before running tests. """
        self.env = Environment(
            T=10, p=0.8, a_ret=0.2, b_ret=-0.3, riskless_ret=0.02, alpha=0.001,
            W_MAX=6000, W_STEP=50, ACTION_STEP=50
        )

    def test_get_next_state(self):
        """ Test wealth transition based on action. """
        w_next = self.env.get_next_state(1000, 500)
        self.assertTrue(850 <= w_next <= 1100)  # Wealth should be within limits

    def test_utility(self):
        """ Test if utility function computes correctly. """
        utility1 = self.env.utility(1000)
        utility2 = self.env.utility(2000)
        self.assertLess(utility1, utility2)  # More wealth should lead to more negative utility

    def test_get_reward(self):
        """ Test if reward is correctly given at terminal state. """
        reward = self.env.get_reward(self.env.T - 1, 1000)
        self.assertNotEqual(reward, 0.0)  # Should only be non-zero at terminal state

        reward = self.env.get_reward(self.env.T - 2, 1000)
        self.assertEqual(reward, 0.0)

class TestAgent(unittest.TestCase):

    def setUp(self):
        """ Initialize the agent before running tests. """
        self.env = Environment(
            T=10, p=0.8, a_ret=0.6, b_ret=-0.3, riskless_ret=0.02, alpha=0.001,
            W_MAX=300, W_STEP=50, ACTION_STEP=50
        )
        self.agent = Agent(self.env, alpha=0.01, gamma=1.0, epsilon_start=0.2, epsilon_end=0.01, decay_rate=0.005, INITIAL_WEALTH=100)

    def test_build_action_index_map(self):
        """ Ensure action index mapping is created correctly. """
        action_map = self.agent._build_action_index_map()
        # print(action_map)
        self.assertIsInstance(action_map, np.ndarray)  # Ensure it's an ndarray
        self.assertGreater(len(action_map), 0)  # Should not be empty
        

    def test_choose_action(self):
        """ Ensure agent selects valid actions. """
        action = self.agent.choose_action(0, 100, 0.1)
        self.assertIn(action, self.env.action_candidates[100])  # Action should be in valid action space

    def test_update_q_table(self):
        """ Ensure Q-table is updated properly. """
        w, x = 100, 50
        reward = 10
        w_next = self.env.get_next_state(w, x)
        Q_before = self.agent.Q.copy()

        self.agent.update_q_table(0, w, x, reward, w_next)
        Q_after = self.agent.Q.copy()

        self.assertTrue(np.any(Q_after != Q_before))  # Ensure Q-table actually updates

class TestTrainer(unittest.TestCase):

    def setUp(self):
        """ Initialize the trainer before running tests. """
        self.env = Environment(
            T=10, p=0.8, a_ret=0.6, b_ret=-0.3, riskless_ret=0.02, alpha=0.001,
            W_MAX=3000, W_STEP=50, ACTION_STEP=50
        )
        self.agent = Agent(self.env, alpha=0.01, gamma=1.0, epsilon_start=0.2, epsilon_end=0.01, decay_rate=0.005, INITIAL_WEALTH=1000)
        self.trainer = Trainer(self.agent, num_episodes=10)  # Small training episodes for quick tests

    def test_training_updates_q_table(self):
        """ Ensure training updates the Q-table. """
        Q_before = self.agent.Q.copy()
        self.trainer.train()
        Q_after = self.agent.Q.copy()

        self.assertTrue(np.any(Q_after != Q_before))  # Ensure Q-table is modified after training

    def test_plot_results(self):
        """ Ensure plotting function runs without errors. """
        try:
            self.trainer.plot_results()
        except Exception as e:
            self.fail(f"plot_results() raised an error: {e}")

if __name__ == "__main__":
    unittest.main()
