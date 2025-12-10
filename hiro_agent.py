import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List

# --- Environment (Unchanged from T1) ---
class KeyDoorEnv:
    def __init__(self, size=10, max_steps=100, seed=None):
        self.size = size
        self.max_steps = max_steps
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.reset()

    def reset(self):
        self.agent_pos = np.array([0, 0], dtype=np.float32)
        self.key_pos = np.array([random.randint(1, self.size-2), random.randint(1, self.size-2)], dtype=np.float32)
        self.door_pos = np.array([self.size-1, self.size-1], dtype=np.float32)
        self.has_key = False
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.agent_pos, self.key_pos, self.door_pos, [self.has_key]])

    def step(self, action):
        self.steps += 1
        x, y = self.agent_pos
        if action == 0 and y < self.size - 1: y += 1
        if action == 1 and y > 0: y -= 1
        if action == 2 and x < self.size - 1: x += 1
        if action == 3 and x > 0: x -= 1
        self.agent_pos = np.array([x, y], dtype=np.float32)

        reward = -0.1
        done = False
        if np.array_equal(self.agent_pos, self.key_pos) and not self.has_key:
            self.has_key = True; reward = 1.0
        if np.array_equal(self.agent_pos, self.door_pos) and self.has_key:
            reward = 10.0; done = True
        if self.steps >= self.max_steps:
            done = True
        return self._get_obs(), reward, done, {}

# ReplayBuffer remains the same as T1
class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.episodes = []
        self.current_episode = []

    def add(self, experience, is_done):
        self.current_episode.append(experience)
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities) if self.priorities else 1.0)
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)
        if is_done:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            if len(self.episodes) > 200:
                self.episodes.pop(0)

    def sample(self, batch_size, rer_prob=0.2):
        priorities = np.array(self.priorities)**self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        if random.random() < rer_prob and len(self.episodes) > 1:
            main_sample_state_embedding = np.mean([s[0] for s in samples], axis=0)
            best_ep_idx = self._get_similar_episode(main_sample_state_embedding)
            retrieved_episode = self.episodes[best_ep_idx]
            extra_samples = random.sample(retrieved_episode, min(len(retrieved_episode), batch_size // 4))
            samples.extend(extra_samples)

        return samples, indices

    def update_priorities(self, indices, new_priorities):
        for i, p in zip(indices, new_priorities): self.priorities[i] = p

    def _get_similar_episode(self, target_embedding):
        similarities = []
        for ep in self.episodes:
            ep_embedding = np.mean([s[0] for s in ep], axis=0)
            # Cosine similarity
            sim = np.dot(target_embedding, ep_embedding) / ((np.linalg.norm(target_embedding) * np.linalg.norm(ep_embedding)) + 1e-8)
            similarities.append(sim)
        return np.argmax(similarities)

# --- T4 Components ---
class T4Config:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 batch_size: int = 64,
                 replay_capacity: int = 10000,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 latent_dim: int = 32,
                 transformer_heads: int = 4,
                 transformer_layers: int = 4,
                 planning_horizon: int = 12):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.replay_capacity = replay_capacity
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.latent_dim = latent_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.planning_horizon = planning_horizon

class Encoder(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, state_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, state_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DiscreteAutoencoder(nn.Module):
    def __init__(self, state_dim, latent_dim, num_embeddings=128):
        super(DiscreteAutoencoder, self).__init__()
        self.encoder = Encoder(state_dim, latent_dim)
        self.decoder = Decoder(latent_dim, state_dim)
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

    def encode(self, x):
        encoded = self.encoder(x)
        distances = torch.sum((encoded.unsqueeze(1) - self.embedding.weight)**2, dim=2)
        return torch.argmin(distances, dim=1)

    def decode(self, z_indices):
        z_quantized = self.embedding(z_indices)
        return self.decoder(z_quantized)

    def forward(self, x):
        z_indices = self.encode(x)
        return self.decode(z_indices)

class DynamicsModel(nn.Module):
    def __init__(self, latent_dim, action_dim, num_embeddings, transformer_layers, transformer_heads):
        super(DynamicsModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim + action_dim, nhead=transformer_heads),
            num_layers=transformer_layers
        )
        self.fc = nn.Linear(latent_dim + action_dim, num_embeddings)

    def forward(self, z, a):
        za = torch.cat([z, a], dim=-1)
        transformer_out = self.transformer(za.unsqueeze(0)).squeeze(0)
        return self.fc(transformer_out)

class RewardPredictor(nn.Module):
    def __init__(self, latent_dim):
        super(RewardPredictor, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        return self.fc2(x)

class Actor(nn.Module):
    def __init__(self, latent_dim, action_dim, planning_horizon):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, action_dim * planning_horizon)
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon

    def forward(self, z):
        actions = self.fc2(F.relu(self.fc1(z)))
        return actions.view(-1, self.planning_horizon, self.action_dim)

class Critic(nn.Module):
    def __init__(self, latent_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, z):
        return self.fc2(F.relu(self.fc1(z)))

class T4_Agent:
    def __init__(self, config: T4Config):
        self.config = config
        self.autoencoder = DiscreteAutoencoder(config.state_dim, config.latent_dim)
        self.dynamics_model = DynamicsModel(config.latent_dim, config.action_dim, self.autoencoder.num_embeddings, config.transformer_layers, config.transformer_heads)
        self.reward_predictor = RewardPredictor(config.latent_dim)
        self.actor = Actor(config.latent_dim, config.action_dim, config.planning_horizon)
        self.critic = Critic(config.latent_dim)
        self.replay_buffer = ReplayBuffer(config.replay_capacity)

        self.world_model_params = list(self.autoencoder.parameters()) + list(self.dynamics_model.parameters()) + list(self.reward_predictor.parameters())
        self.world_model_optimizer = optim.Adam(self.world_model_params, lr=config.learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)

    def train_world_model(self):
        if len(self.replay_buffer.buffer) < self.config.batch_size:
            return

        samples, _ = self.replay_buffer.sample(self.config.batch_size)
        s, a, r, s_next, done = map(np.array, zip(*samples))

        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s_next = torch.FloatTensor(s_next)

        # Autoencoder loss
        s_recon = self.autoencoder(s)
        recon_loss = F.mse_loss(s_recon, s)

        # Dynamics and reward loss
        z_indices = self.autoencoder.encode(s)
        z = self.autoencoder.embedding(z_indices)
        a_one_hot = F.one_hot(a, num_classes=self.config.action_dim).float()

        z_next_pred_logits = self.dynamics_model(z, a_one_hot)
        z_next_indices_true = self.autoencoder.encode(s_next)
        dynamics_loss = F.cross_entropy(z_next_pred_logits, z_next_indices_true)

        r_pred = self.reward_predictor(z)
        reward_loss = F.mse_loss(r_pred.squeeze(), r)

        total_loss = recon_loss + dynamics_loss + reward_loss
        self.world_model_optimizer.zero_grad()
        total_loss.backward()
        self.world_model_optimizer.step()

    def train_actor_critic(self):
        if len(self.replay_buffer.buffer) < self.config.batch_size:
            return

        samples, _ = self.replay_buffer.sample(self.config.batch_size)
        s, _, _, _, _ = map(np.array, zip(*samples))
        s = torch.FloatTensor(s)

        z_indices_start = self.autoencoder.encode(s)
        z_start = self.autoencoder.embedding(z_indices_start)

        # Imagine trajectories
        z = z_start
        imagined_rewards = []
        imagined_latents = []
        actions = self.actor(z_start)

        for t in range(self.config.planning_horizon):
            action_t = actions[:, t, :]
            z_next_logits = self.dynamics_model(z, action_t)
            z_next_indices = torch.argmax(z_next_logits, dim=-1)
            z_next = self.autoencoder.embedding(z_next_indices)

            reward = self.reward_predictor(z_next)
            imagined_rewards.append(reward)
            imagined_latents.append(z_next)
            z = z_next

        # Actor loss
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        imagined_latents = torch.stack(imagined_latents, dim=1)
        values = self.critic(imagined_latents)

        returns = imagined_rewards + self.config.gamma * values
        actor_loss = -torch.mean(returns)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Critic loss
        critic_loss = F.mse_loss(self.critic(z_start.detach()), returns.mean().detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        z_indices = self.autoencoder.encode(state)
        z = self.autoencoder.embedding(z_indices)
        actions = self.actor(z)
        return torch.argmax(actions[0, 0, :]).item()

def run_evaluation(agent: T4_Agent, envs: List[KeyDoorEnv], num_episodes: int = 150):
    episodes_to_solve = []

    for env in envs:
        solved = False
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.replay_buffer.add((state, action, reward, next_state, done), done)
                state = next_state

                if done and reward > 5:
                    solved = True

            agent.train_world_model()
            agent.train_actor_critic()

            if solved:
                episodes_to_solve.append(episode)
                break

    avg_episodes = np.mean(episodes_to_solve) if episodes_to_solve else float('inf')
    return avg_episodes

if __name__ == '__main__':
    envs = [KeyDoorEnv(seed=i) for i in range(4)]
    env_sample = envs[0]
    config = T4Config(state_dim=env_sample._get_obs().shape[0],
                      action_dim=4)
    agent = T4_Agent(config)

    avg_episodes = run_evaluation(agent, envs)
    print(f"T4 Agent - Average episodes to first solve: {avg_episodes:.2f}")
