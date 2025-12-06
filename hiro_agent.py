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

from typing import List, Tuple, Dict, Any

class T4Config:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 size: int,
                 latent_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_decay: float = 0.999,
                 epsilon_min: float = 0.05,
                 lr: float = 1e-4,
                 replay_capacity: int = 10000,
                 batch_size: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.size = size
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size

class DiscreteAutoencoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, num_embeddings: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_e = self.encoder(x)
        distances = (z_e.unsqueeze(1) - self.embedding.weight).pow(2).sum(2)
        _, z_q_indices = distances.min(1)
        return z_q_indices

    def decode(self, z_q_indices: torch.Tensor) -> torch.Tensor:
        z_q = self.embedding(z_q_indices)
        return self.decoder(z_q)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        distances = (z_e.unsqueeze(1) - self.embedding.weight).pow(2).sum(2)
        _, z_q_indices = distances.min(1)
        z_q = self.embedding(z_q_indices)
        x_recon = self.decoder(z_q)
        return x_recon, z_e, z_q

class DynamicsModel(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, a], dim=-1))

class RewardPredictor(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class T4_Agent:
    def __init__(self, config: T4Config):
        self.config = config
        self.autoencoder = DiscreteAutoencoder(config.state_dim, config.latent_dim, 512)
        self.dynamics_model = DynamicsModel(config.latent_dim, config.action_dim)
        self.reward_predictor = RewardPredictor(config.latent_dim)
        self.actor = Actor(config.latent_dim, config.action_dim)
        self.critic = Critic(config.latent_dim)
        self.replay_buffer = ReplayBuffer(config.replay_capacity)

        self.dynamics_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=config.lr)
        self.reward_optimizer = optim.Adam(self.reward_predictor.parameters(), lr=config.lr)
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=config.lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)

    def choose_action(self, state: np.ndarray) -> int:
        if random.random() < self.config.epsilon:
            return random.randint(0, self.config.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            latent_indices = self.autoencoder.encode(state_tensor)
            latent_state = self.autoencoder.embedding(latent_indices)
            action_probs = self.actor(latent_state)
        return action_probs.argmax().item()

    def train(self):
        if len(self.replay_buffer.buffer) < self.config.batch_size:
            return

        self._train_world_model()
        self._train_actor_critic()

        self.config.epsilon = max(self.config.epsilon_min, self.config.epsilon * self.config.epsilon_decay)

    def _train_world_model(self):
        samples, _ = self.replay_buffer.sample(self.config.batch_size)
        s, a, r, s_next, done = map(np.array, zip(*samples))

        s_tensor = torch.FloatTensor(s)
        a_tensor = torch.LongTensor(a)
        r_tensor = torch.FloatTensor(r)
        s_next_tensor = torch.FloatTensor(s_next)

        # Train Autoencoder
        s_recon, z_e, z_q = self.autoencoder(s_tensor)
        recon_loss = F.mse_loss(s_recon, s_tensor)
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        autoencoder_loss = recon_loss + commitment_loss

        self.autoencoder_optimizer.zero_grad()
        autoencoder_loss.backward()
        self.autoencoder_optimizer.step()

        # Train Dynamics and Reward Models
        with torch.no_grad():
            latent_indices = self.autoencoder.encode(s_tensor)
            latent_state = self.autoencoder.embedding(latent_indices)
            next_latent_indices = self.autoencoder.encode(s_next_tensor)
            next_latent_state = self.autoencoder.embedding(next_latent_indices)

        # Dynamics Model
        a_one_hot = F.one_hot(a_tensor, num_classes=self.config.action_dim).float()
        next_latent_pred = self.dynamics_model(latent_state, a_one_hot)
        dynamics_loss = F.mse_loss(next_latent_pred, next_latent_state)

        self.dynamics_optimizer.zero_grad()
        dynamics_loss.backward()
        self.dynamics_optimizer.step()

        # Reward Predictor
        reward_pred = self.reward_predictor(latent_state)
        reward_loss = F.mse_loss(reward_pred, r_tensor.unsqueeze(1))

        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()

    def _train_actor_critic(self):
        # Sample initial states from replay buffer
        samples, _ = self.replay_buffer.sample(self.config.batch_size)
        s, _, _, _, _ = map(np.array, zip(*samples))
        s_tensor = torch.FloatTensor(s)

        # Encode initial states
        with torch.no_grad():
            latent_indices = self.autoencoder.encode(s_tensor)
            latent_state = self.autoencoder.embedding(latent_indices)

        # Imagine trajectories
        imagined_latents = []
        imagined_actions = []
        current_latent = latent_state

        for _ in range(15): # Imagination horizon
            action_probs = self.actor(current_latent)
            action = F.gumbel_softmax(action_probs, hard=True)
            next_latent = self.dynamics_model(current_latent, action)

            imagined_latents.append(current_latent)
            imagined_actions.append(action)

            current_latent = next_latent

        imagined_latents = torch.stack(imagined_latents)
        imagined_actions = torch.stack(imagined_actions)

        # Predict rewards and values
        imagined_rewards = self.reward_predictor(imagined_latents)
        imagined_values = self.critic(imagined_latents)

        # Calculate targets
        returns = torch.zeros_like(imagined_rewards)
        last_val = imagined_values[-1]
        for t in reversed(range(14)):
            returns[t] = imagined_rewards[t] + self.config.gamma * last_val
            last_val = returns[t]

        # Update Critic
        critic_loss = F.mse_loss(imagined_values[:-1], returns[:-1].detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -torch.mean(returns[:-1])
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


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
                agent.train()
                if reward > 5:
                    solved = True
            if solved:
                episodes_to_solve.append(episode)
                break
    avg_episodes = np.mean(episodes_to_solve) if episodes_to_solve else float('inf')
    return avg_episodes


if __name__ == '__main__':
    envs = [KeyDoorEnv(seed=i) for i in range(4)]
    env_sample = envs[0]
    config = T4Config(state_dim=env_sample._get_obs().shape[0],
                      action_dim=4,
                      size=env_sample.size)
    agent = T4_Agent(config)

    avg_episodes = run_evaluation(agent, envs)
    print(f"T4 Agent - Average episodes to first solve: {avg_episodes:.2f}")
