import unittest
import torch
from hiro_agent import T4Config, DiscreteAutoencoder, DynamicsModel, RewardPredictor, Actor, Critic

class TestHiroAgentT4(unittest.TestCase):

    def setUp(self):
        self.config = T4Config(state_dim=8, action_dim=4, size=10, latent_dim=32)

    def test_autoencoder(self):
        autoencoder = DiscreteAutoencoder(self.config.state_dim, self.config.latent_dim, 512)
        state = torch.randn(1, self.config.state_dim)
        recon_state, _, _ = autoencoder(state)
        self.assertEqual(recon_state.shape, state.shape)

    def test_dynamics_model(self):
        dynamics_model = DynamicsModel(self.config.latent_dim, self.config.action_dim)
        latent_state = torch.randn(1, self.config.latent_dim)
        action = torch.randn(1, self.config.action_dim)
        next_latent_state = dynamics_model(latent_state, action)
        self.assertEqual(next_latent_state.shape, latent_state.shape)

    def test_reward_predictor(self):
        reward_predictor = RewardPredictor(self.config.latent_dim)
        latent_state = torch.randn(1, self.config.latent_dim)
        reward = reward_predictor(latent_state)
        self.assertEqual(reward.shape, (1, 1))

    def test_actor_critic(self):
        actor = Actor(self.config.latent_dim, self.config.action_dim)
        critic = Critic(self.config.latent_dim)
        latent_state = torch.randn(1, self.config.latent_dim)

        action = actor(latent_state)
        value = critic(latent_state)

        self.assertEqual(action.shape, (1, self.config.action_dim))
        self.assertEqual(value.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
