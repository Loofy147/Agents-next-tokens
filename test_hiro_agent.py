import unittest
import torch
import numpy as np
from hiro_agent import KeyDoorEnv, T4Config, T4_Agent, ReplayBuffer

class TestT4Agent(unittest.TestCase):
    def setUp(self):
        self.env = KeyDoorEnv(size=5, max_steps=10)
        obs_dim = self.env._get_obs().shape[0]
        self.config = T4Config(state_dim=obs_dim, action_dim=4, size=5,
                              latent_dim=8, codebook_size=16,
                              transformer_layers=1, transformer_heads=1,
                              transformer_dim=32, batch_size=4,
                              imagination_horizon=2)
        self.agent = T4_Agent(self.config)

    def test_choose_action(self):
        state = self.env.reset()
        action = self.agent.choose_action(state, train=False)
        self.assertIn(action, [0, 1, 2, 3])

    def test_train_step(self):
        # Fill replay buffer
        for _ in range(self.config.batch_size):
            s = self.env.reset()
            a = 0
            ns, r, d, _ = self.env.step(a)
            self.agent.replay.add((s, a, r, ns, d))

        # This should run without error
        self.agent.train()

    def test_env_interaction(self):
        state = self.env.reset()
        next_state, reward, done, info = self.env.step(0)
        self.assertEqual(next_state.shape, (7,))

if __name__ == "__main__":
    unittest.main()
