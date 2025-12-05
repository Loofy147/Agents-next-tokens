import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple

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

# --- T2 Components ---
class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, dropout_rate: float = 0.1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class InverseModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_goal_dim=8):
        super(InverseModel, self).__init__()
        self.fc1 = nn.Linear(state_dim * 2, 64)
        self.fc2 = nn.Linear(64, latent_goal_dim)
        self.fc3 = nn.Linear(latent_goal_dim, action_dim)

    def forward(self, s, s_next):
        x = torch.cat([s, s_next], dim=-1)
        x = F.relu(self.fc1(x))
        latent_goal = self.fc2(x)
        action_pred = self.fc3(F.relu(latent_goal))
        return action_pred, latent_goal

class Empowerment(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # State-action encoder
        self.sa_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)  # Embedding dim
        )

        # Next-state encoder
        self.ns_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64)
        )

        self.temperature = 0.1

    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, batch_negatives: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        InfoNCE: I(a; s') ≈ log(exp(f(a,s'))/sum(exp(f(a,s'_i))))
        """
        # Positive pair
        sa_embed = self.sa_encoder(torch.cat([state, action], dim=-1))
        ns_embed = self.ns_encoder(next_state)

        # Cosine similarity
        pos_sim = F.cosine_similarity(sa_embed, ns_embed) / self.temperature

        if batch_negatives:
            # Use other batch samples as negatives (efficiency trick)
            neg_sim = torch.mm(sa_embed, ns_embed.T) / self.temperature
            neg_sim.fill_diagonal_(-float('inf'))

            # InfoNCE loss
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits, labels)

            # Empowerment reward = positive similarity (higher = more control)
            empowerment_reward = pos_sim.detach()
        else:
            empowerment_reward = pos_sim
            loss = -pos_sim.mean()

        return empowerment_reward, loss


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

class T3Config:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 size: int,
                 latent_goal_dim: int = 8,
                 manager_interval: int = 10,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_decay: float = 0.999,
                 epsilon_min: float = 0.05,
                 cql_penalty: float = 0.1,
                 empowerment_coeff: float = 0.2,
                 infogain_coeff: float = 0.1,
                 base_augmentation_noise: float = 0.02,
                 worker_lr: float = 1e-3,
                 manager_lr: float = 1e-3,
                 inverse_model_lr: float = 1e-3,
                 empowerment_lr: float = 1e-3,
                 worker_replay_capacity: int = 2000,
                 manager_replay_capacity: int = 200,
                 batch_size: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.size = size
        self.latent_goal_dim = latent_goal_dim
        self.manager_interval = manager_interval
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.cql_penalty = cql_penalty
        self.empowerment_coeff = empowerment_coeff
        self.infogain_coeff = infogain_coeff
        self.base_augmentation_noise = base_augmentation_noise
        self.worker_lr = worker_lr
        self.manager_lr = manager_lr
        self.inverse_model_lr = inverse_model_lr
        self.empowerment_lr = empowerment_lr
        self.worker_replay_capacity = worker_replay_capacity
        self.manager_replay_capacity = manager_replay_capacity
        self.batch_size = batch_size

# --- T3 Hierarchical Agent ---
class T3_Agent:
    def __init__(self, config: T3Config):
        self.config = config

        # Goal-conditioned Inverse Model
        self.inverse_model = InverseModel(config.state_dim, config.action_dim, config.latent_goal_dim)
        self.inverse_model_optimizer = optim.Adam(self.inverse_model.parameters(), lr=config.inverse_model_lr)

        # Empowerment Module
        self.empowerment = Empowerment(config.state_dim, config.action_dim)
        self.empowerment_optimizer = optim.Adam(self.empowerment.parameters(), lr=config.empowerment_lr)

        # Worker (conditioned on latent goal)
        self.worker_critics = [Critic(config.state_dim + config.latent_goal_dim, config.action_dim) for _ in range(3)]
        self.worker_targets = [Critic(config.state_dim + config.latent_goal_dim, config.action_dim) for _ in range(3)]
        self.worker_optimizers = [optim.Adam(c.parameters(), lr=config.worker_lr) for c in self.worker_critics]
        self.worker_replay = ReplayBuffer(config.worker_replay_capacity)

        # Manager (action space is selecting a direction vector)
        self.manager_actions = [np.array([0, 3]), np.array([0, -3]), np.array([3, 0]), np.array([-3, 0]),
                                np.array([2, 2]), np.array([2, -2]), np.array([-2, 2]), np.array([-2, -2])]
        self.manager_action_dim = len(self.manager_actions)
        self.manager_critics = [Critic(config.state_dim, self.manager_action_dim) for _ in range(3)]
        self.manager_targets = [Critic(config.state_dim, self.manager_action_dim) for _ in range(3)]
        self.manager_optimizers = [optim.Adam(c.parameters(), lr=config.manager_lr) for c in self.manager_critics]
        self.manager_replay = ReplayBuffer(config.manager_replay_capacity)

    def get_latent_goal(self, s: np.ndarray, s_g: np.ndarray) -> np.ndarray:
        s_t = torch.FloatTensor(s).unsqueeze(0)
        s_g_t = torch.FloatTensor(s_g).unsqueeze(0)
        with torch.no_grad():
            _, latent_goal = self.inverse_model(s_t, s_g_t)
        return latent_goal.squeeze(0).numpy()

    def choose_manager_action(self, state: np.ndarray) -> int:
        if random.random() < self.config.epsilon:
            return random.randint(0, self.manager_action_dim - 1)
        with torch.no_grad():
            q_values = torch.mean(torch.stack([c(torch.FloatTensor(state)) for c in self.manager_critics]), dim=0)
        return q_values.argmax().item()

    def choose_action(self, state: np.ndarray, latent_goal: np.ndarray) -> int:
        if random.random() < self.config.epsilon: return random.randint(0, self.config.action_dim - 1)
        state_goal = torch.FloatTensor(np.concatenate([state, latent_goal]))
        with torch.no_grad():
            q_values = torch.mean(torch.stack([c(state_goal) for c in self.worker_critics]), dim=0)
        return q_values.argmax().item()

    def train(self):
        self._train_intrinsic_modules()
        self._train_worker()
        self._train_manager()
        self.config.epsilon = max(self.config.epsilon_min, self.config.epsilon * self.config.epsilon_decay)

    def _train_intrinsic_modules(self):
        if len(self.worker_replay.buffer) < self.config.batch_size: return
        self._train_inverse_model(self.config.batch_size)
        self._train_empowerment(self.config.batch_size)

    def _train_empowerment(self, batch_size: int):
        if len(self.worker_replay.buffer) < batch_size: return
        samples, _ = self.worker_replay.sample(batch_size)
        s, a, _, s_next, _, _ = map(np.array, zip(*samples))

        s = torch.FloatTensor(s)
        a = F.one_hot(torch.LongTensor(a), num_classes=self.config.action_dim).float()
        s_next = torch.FloatTensor(s_next)

        _, loss = self.empowerment(s, a, s_next)

        self.empowerment_optimizer.zero_grad()
        loss.backward()
        self.empowerment_optimizer.step()

    def _train_inverse_model(self, batch_size: int):
        if len(self.worker_replay.buffer) < batch_size: return
        samples, _ = self.worker_replay.sample(batch_size)
        s, a, _, s_next, _, _ = map(np.array, zip(*samples))

        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        s_next = torch.FloatTensor(s_next)

        action_pred, _ = self.inverse_model(s, s_next)
        loss = F.cross_entropy(action_pred, a)

        self.inverse_model_optimizer.zero_grad()
        loss.backward()
        self.inverse_model_optimizer.step()

    def _train_worker(self):
        if len(self.worker_replay.buffer) < self.config.batch_size: return
        samples, indices = self.worker_replay.sample(self.config.batch_size)

        # Latent Goal Relabeling (HER)
        processed_samples = []
        for s, a, r, s_next, done, z in samples:
            if random.random() < 0.8:
                z_achieved = self.get_latent_goal(s, s_next)
                r_new = 0.0 # High reward for achieving the relabeled goal
                processed_samples.append((s, a, r_new, s_next, done, z_achieved))
            else:
                processed_samples.append((s, a, r, s_next, done, z))

        s, a, r, s_next, done, z = map(np.array, zip(*processed_samples))
        s_tensor = torch.FloatTensor(s)
        a_long = torch.LongTensor(a)
        r_tensor = torch.FloatTensor(r)
        s_next_tensor = torch.FloatTensor(s_next)
        done_tensor = torch.FloatTensor(done)
        z_tensor = torch.FloatTensor(z)

        # Adaptive noise augmentation
        td_errors = self._calculate_td_error(s_tensor, z_tensor, a_long, r_tensor, s_next_tensor, done_tensor)
        alpha = 0.5
        td_normalized = (td_errors - td_errors.mean()) / (td_errors.std() + 1e-8)
        noise_scale = self.config.base_augmentation_noise * (1 + alpha * td_normalized.unsqueeze(1))
        s_aug = s_tensor + torch.randn_like(s_tensor) * noise_scale

        a_long = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s_next = torch.FloatTensor(s_next)
        done = torch.FloatTensor(done)
        z = torch.FloatTensor(z)

        # --- T3: Intrinsic Rewards ---
        r = self._compute_intrinsic_rewards(s_aug, a_long, s_next_tensor, z_tensor, r_tensor)

        # Critic training
        td_errors = self._update_worker_critics(s_aug, z_tensor, a_long, r, s_next_tensor, done_tensor)
        self.worker_replay.update_priorities(indices, np.mean(td_errors, axis=0).flatten())
        self._update_targets(self.worker_critics, self.worker_targets)

    def _calculate_td_error(self, s: torch.Tensor, z: torch.Tensor, a_long: torch.Tensor, r: torch.Tensor, s_next: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sg = torch.cat([s, z], dim=1)
            s_next_g = torch.cat([s_next, z], dim=1)
            avg_next_q = torch.mean(torch.stack([t(s_next_g) for t in self.worker_targets]), dim=0)
            greedy_a = avg_next_q.argmax(dim=1, keepdim=True)
            min_q_next = torch.min(torch.stack([t(s_next_g).gather(1, greedy_a) for t in self.worker_targets]), dim=0)[0]
            target = r.unsqueeze(1) + self.config.gamma * (1 - done.unsqueeze(1)) * (min_q_next - self.config.cql_penalty)

            q_values = torch.stack([c(sg).gather(1, a_long.unsqueeze(1)) for c in self.worker_critics])
            td_errors = (q_values - target).abs().mean(dim=0).squeeze()
        return td_errors

    def _compute_epistemic_uncertainty(self, state: torch.Tensor, latent_goal: torch.Tensor, n_samples: int = 5) -> torch.Tensor:
        """
        Epistemic uncertainty via MC dropout
        """
        for critic in self.worker_critics:
            critic.train()  # Enable dropout

        q_samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                q_preds = torch.stack([
                    c(torch.cat([state, latent_goal], dim=1))
                    for c in self.worker_critics
                ])
                q_samples.append(q_preds)

        # Variance across MC samples (epistemic)
        q_samples_tensor = torch.stack(q_samples)  # [n_samples, n_critics, batch, actions]
        epistemic_unc = q_samples_tensor.var(dim=0).mean(dim=0).max(dim=1)[0]  # Per-state

        for critic in self.worker_critics:
            critic.eval() # Disable dropout
        return epistemic_unc

    def _compute_intrinsic_rewards(self, s: torch.Tensor, a_long: torch.Tensor, s_next: torch.Tensor, z: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # Empowerment Reward
        empowerment_reward, _ = self.empowerment(s, F.one_hot(a_long, num_classes=self.config.action_dim).float(), s_next)

        # Info-Gain Reward (critic disagreement)
        info_gain_reward = self._compute_epistemic_uncertainty(s, z)

        intrinsic_reward = self.config.empowerment_coeff * empowerment_reward + self.config.infogain_coeff * info_gain_reward
        return r + intrinsic_reward.detach()

    def _update_worker_critics(self, s: torch.Tensor, z: torch.Tensor, a_long: torch.Tensor, r: torch.Tensor, s_next: torch.Tensor, done: torch.Tensor) -> List[np.ndarray]:
        td_errors = []
        for i in range(3):
            sg = torch.cat([s, z], dim=1)
            s_next_g = torch.cat([s_next, z], dim=1)
            with torch.no_grad():
                avg_next_q = torch.mean(torch.stack([t(s_next_g) for t in self.worker_targets]), dim=0)
                greedy_a = avg_next_q.argmax(dim=1, keepdim=True)
                min_q_next = torch.min(torch.stack([t(s_next_g).gather(1, greedy_a) for t in self.worker_targets]), dim=0)[0]
                target = r.unsqueeze(1) + self.config.gamma * (1 - done.unsqueeze(1)) * (min_q_next - self.config.cql_penalty)

            q_values = self.worker_critics[i](sg).gather(1, a_long.unsqueeze(1))
            loss = F.mse_loss(q_values, target)
            self.worker_optimizers[i].zero_grad()
            loss.backward()
            self.worker_optimizers[i].step()
            td_errors.append((q_values - target).abs().detach().numpy())
        return td_errors

    def _train_manager(self):
        if len(self.manager_replay.buffer) < self.config.batch_size: return
        samples, indices = self.manager_replay.sample(self.config.batch_size, rer_prob=0.0)
        s, a_k, r_sum, s_next, done, _ = map(np.array, zip(*samples))

        s = torch.FloatTensor(s)
        a_k = torch.LongTensor(a_k)
        r_sum = torch.FloatTensor(r_sum)
        s_next = torch.FloatTensor(s_next)
        done = torch.FloatTensor(done)

        td_errors = []
        for i in range(3):
            with torch.no_grad():
                q_next = torch.mean(torch.stack([t(s_next) for t in self.manager_targets]), dim=0).max(1)[0]
                target = r_sum + self.config.gamma * (1 - done) * q_next
            q_values = self.manager_critics[i](s).gather(1, a_k.unsqueeze(1))
            loss = F.mse_loss(q_values, target.unsqueeze(1))
            self.manager_optimizers[i].zero_grad()
            loss.backward()
            self.manager_optimizers[i].step()
            td_errors.append((q_values - target.unsqueeze(1)).abs().detach().numpy())
        self.manager_replay.update_priorities(indices, np.mean(td_errors, axis=0).flatten())
        self._update_targets(self.manager_critics, self.manager_targets)

    def _update_targets(self, critics: List[nn.Module], targets: List[nn.Module]):
        for i in range(3):
            for target_param, param in zip(targets[i].parameters(), critics[i].parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)

def run_evaluation(agent: T3_Agent, envs: List[KeyDoorEnv], num_episodes: int = 150):
    total_hits, total_manager_steps = 0, 0
    episodes_to_solve = []

    for env in envs:
        solved = False
        for episode in range(num_episodes):
            state = env.reset()
            manager_state, cumulative_r, ep_done = state, 0, False

            while not ep_done:
                manager_action_idx = agent.choose_manager_action(manager_state)
                direction_vector = agent.manager_actions[manager_action_idx]

                goal_state = manager_state.copy()
                goal_state[:2] += direction_vector
                goal_state[:2] = np.clip(goal_state[:2], 0, agent.config.size - 1)

                latent_goal = agent.get_latent_goal(manager_state, goal_state)

                achieved_state = None
                for _ in range(agent.config.manager_interval):
                    action = agent.choose_action(state, latent_goal)
                    next_state, reward, done, _ = env.step(action)

                    worker_reward = -np.linalg.norm(next_state[:2] - goal_state[:2])
                    agent.worker_replay.add((state, action, worker_reward, next_state, done, latent_goal), done)

                    state = next_state
                    cumulative_r += reward
                    achieved_state = next_state
                    if done:
                        if reward > 5: solved = True
                        ep_done = True
                        break

                agent.manager_replay.add((manager_state, manager_action_idx, cumulative_r, state, ep_done, goal_state), ep_done)
                if np.linalg.norm(achieved_state[:2] - goal_state[:2]) < 1.5:
                    total_hits += 1
                total_manager_steps += 1

                manager_state, cumulative_r = state, 0
                agent.train()

            if solved:
                episodes_to_solve.append(episode)
                break

    avg_episodes = np.mean(episodes_to_solve) if episodes_to_solve else float('inf')
    hit_rate = total_hits / total_manager_steps if total_manager_steps > 0 else 0
    return avg_episodes, hit_rate

if __name__ == '__main__':
    envs = [KeyDoorEnv(seed=i) for i in range(4)]
    env_sample = envs[0]
    config = T3Config(state_dim=env_sample._get_obs().shape[0],
                      action_dim=4,
                      size=env_sample.size)
    agent = T3_Agent(config)

    avg_episodes, hit_rate = run_evaluation(agent, envs)
    print(f"T3 Agent - Average episodes to first solve: {avg_episodes:.2f}")
    print(f"T3 Agent - Subgoal hit rate: {hit_rate:.2%}")
