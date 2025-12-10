import unittest
import torch
import numpy as np
from hiro_agent import T4Config, DiscreteAutoencoder, DynamicsModel, Actor, Critic

class TestHiroAgentT4(unittest.TestCase):

    def setUp(self):
        self.config = T4Config(state_dim=8, action_dim=4)

    def test_autoencoder(self):
        autoencoder = DiscreteAutoencoder(self.config.state_dim, self.config.latent_dim)
        state = torch.randn(1, self.config.state_dim)

        # Test encode
        encoded_indices = autoencoder.encode(state)
        self.assertEqual(encoded_indices.shape, (1,))

        # Test decode
        decoded_state = autoencoder.decode(encoded_indices)
        self.assertEqual(decoded_state.shape, (1, self.config.state_dim))

        # Test forward pass
        reconstructed_state = autoencoder(state)
        self.assertEqual(reconstructed_state.shape, (1, self.config.state_dim))

    def test_dynamics_model(self):
        autoencoder = DiscreteAutoencoder(self.config.state_dim, self.config.latent_dim)
        dynamics = DynamicsModel(self.config.latent_dim, self.config.action_dim, autoencoder.num_embeddings, self.config.transformer_layers, self.config.transformer_heads)

        z = torch.randn(1, self.config.latent_dim)
        a = torch.randn(1, self.config.action_dim)

        next_z_logits = dynamics(z, a)
        self.assertEqual(next_z_logits.shape, (1, autoencoder.num_embeddings))

    def test_actor_critic(self):
        actor = Actor(self.config.latent_dim, self.config.action_dim, self.config.planning_horizon)
        critic = Critic(self.config.latent_dim)

        z = torch.randn(1, self.config.latent_dim)

        # Test actor
        actions = actor(z)
        self.assertEqual(actions.shape, (1, self.config.planning_horizon, self.config.action_dim))

        # Test critic
        value = critic(z)
        self.assertEqual(value.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
