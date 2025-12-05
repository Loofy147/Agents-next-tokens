import unittest
import torch
import numpy as np
from hiro_agent import T3Config, Empowerment, Critic, SubgoalDiffuser

class TestHiroAgent(unittest.TestCase):

    def setUp(self):
        self.config = T3Config(state_dim=8, action_dim=4, size=10)

    def test_empowerment_module(self):
        empowerment = Empowerment(self.config.state_dim, self.config.action_dim)
        state = torch.randn(1, self.config.state_dim)
        action = torch.randn(1, self.config.action_dim)
        next_state = torch.randn(1, self.config.state_dim)

        empowerment_reward, loss = empowerment(state, action, next_state, batch_negatives=False)
        self.assertEqual(empowerment_reward.shape, (1,))
        self.assertIsNotNone(loss)

    def test_critic_network(self):
        critic = Critic(self.config.state_dim, self.config.action_dim)
        state = torch.randn(1, self.config.state_dim)
        q_values = critic(state)
        self.assertEqual(q_values.shape, (1, self.config.action_dim))

    def test_subgoal_diffuser(self):
        diffuser = SubgoalDiffuser(self.config.state_dim, self.config.latent_goal_dim)
        state = torch.randn(1, self.config.state_dim)
        subgoal = diffuser.sample(state)
        self.assertEqual(subgoal.shape, (1, self.config.latent_goal_dim))

if __name__ == '__main__':
    unittest.main()
