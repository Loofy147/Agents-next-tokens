import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, act_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

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
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Empowerment, self).__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, state, action, next_state):
        # Empowerment as mutual information I(a; s')
        predicted_next_state = self.forward_model(torch.cat([state, action], dim=-1))
        # For simplicity, we'll use a noise-contrastive estimation proxy
        real_pair = torch.cat([state, next_state], dim=-1)
        fake_pair = torch.cat([state, predicted_next_state], dim=-1)

        real_score = self.discriminator(real_pair)
        fake_score = self.discriminator(fake_pair)

        # This is a simplified proxy for mutual information
        empowerment_reward = -torch.log(1 - real_score + 1e-8) + torch.log(fake_score + 1e-8)
        return empowerment_reward


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

# --- T3 Hierarchical Agent ---
class T3_Agent:
    def __init__(self, state_dim, action_dim, size, latent_goal_dim=8):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.size = size
        self.latent_goal_dim = latent_goal_dim
        self.manager_interval = 10
        self.tau = 0.005
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.cql_penalty = 0.1
        self.empowerment_coeff = 0.2
        self.infogain_coeff = 0.1
        self.augmentation_noise = 0.02


        # Goal-conditioned Inverse Model
        self.inverse_model = InverseModel(state_dim, action_dim, latent_goal_dim)
        self.inverse_model_optimizer = optim.Adam(self.inverse_model.parameters(), lr=1e-3)

        # Empowerment Module
        self.empowerment = Empowerment(state_dim, action_dim)
        self.empowerment_optimizer = optim.Adam(self.empowerment.parameters(), lr=1e-3)

        # Worker (conditioned on latent goal)
        self.worker_critics = [Critic(state_dim + latent_goal_dim, action_dim) for _ in range(3)]
        self.worker_targets = [Critic(state_dim + latent_goal_dim, action_dim) for _ in range(3)]
        self.worker_optimizers = [optim.Adam(c.parameters(), lr=1e-3) for c in self.worker_critics]
        self.worker_replay = ReplayBuffer(2000)

        # Manager (action space is selecting a direction vector)
        self.manager_actions = [np.array([0, 3]), np.array([0, -3]), np.array([3, 0]), np.array([-3, 0]),
                                np.array([2, 2]), np.array([2, -2]), np.array([-2, 2]), np.array([-2, -2])]
        self.manager_action_dim = len(self.manager_actions)
        self.manager_critics = [Critic(state_dim, self.manager_action_dim) for _ in range(3)]
        self.manager_targets = [Critic(state_dim, self.manager_action_dim) for _ in range(3)]
        self.manager_optimizers = [optim.Adam(c.parameters(), lr=1e-3) for c in self.manager_critics]
        self.manager_replay = ReplayBuffer(200)

    def get_latent_goal(self, s, s_g):
        s_t = torch.FloatTensor(s).unsqueeze(0)
        s_g_t = torch.FloatTensor(s_g).unsqueeze(0)
        with torch.no_grad():
            _, latent_goal = self.inverse_model(s_t, s_g_t)
        return latent_goal.squeeze(0).numpy()

    def choose_manager_action(self, state, trajectory=None):
        if random.random() < self.epsilon:
            return random.randint(0, self.manager_action_dim - 1)
        with torch.no_grad():
            q_values = torch.mean(torch.stack([c(torch.FloatTensor(state)) for c in self.manager_critics]), dim=0)
        return q_values.argmax().item()

    def choose_action(self, state, latent_goal):
        if random.random() < self.epsilon: return random.randint(0, self.action_dim - 1)
        state_goal = torch.FloatTensor(np.concatenate([state, latent_goal]))
        with torch.no_grad():
            q_values = torch.mean(torch.stack([c(state_goal) for c in self.worker_critics]), dim=0)
        return q_values.argmax().item()

    def train(self, batch_size=64):
        self._train_inverse_model(batch_size)
        self._train_worker(batch_size)
        self._train_manager(batch_size)
        self._train_empowerment(batch_size)
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)

    def _train_empowerment(self, batch_size):
        if len(self.worker_replay.buffer) < batch_size: return
        samples, _ = self.worker_replay.sample(batch_size)
        s, a, _, s_next, _, _ = map(np.array, zip(*samples))

        s = torch.FloatTensor(s)
        a = F.one_hot(torch.LongTensor(a), num_classes=self.action_dim).float()
        s_next = torch.FloatTensor(s_next)

        # Train the forward model
        predicted_next_state = self.empowerment.forward_model(torch.cat([s, a], dim=-1))
        forward_loss = F.mse_loss(predicted_next_state, s_next)

        # Train the discriminator
        real_pair = torch.cat([s, s_next], dim=-1)
        fake_pair = torch.cat([s, predicted_next_state.detach()], dim=-1)

        real_score = self.empowerment.discriminator(real_pair)
        fake_score = self.empowerment.discriminator(fake_pair)

        discriminator_loss = F.binary_cross_entropy(real_score, torch.ones_like(real_score)) + \
                             F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))

        empowerment_loss = forward_loss + discriminator_loss
        self.empowerment_optimizer.zero_grad()
        empowerment_loss.backward()
        self.empowerment_optimizer.step()

    def _train_inverse_model(self, batch_size):
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

    def _train_worker(self, batch_size):
        if len(self.worker_replay.buffer) < batch_size: return
        samples, indices = self.worker_replay.sample(batch_size)

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
        s = torch.FloatTensor(s) + torch.randn_like(torch.FloatTensor(s)) * self.augmentation_noise
        a_long = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s_next = torch.FloatTensor(s_next)
        done = torch.FloatTensor(done)
        z = torch.FloatTensor(z)

        # --- T3: Intrinsic Rewards ---
        # Empowerment Reward
        empowerment_reward = self.empowerment(s, F.one_hot(a_long, num_classes=self.action_dim).float(), s_next).squeeze(-1)

        # Info-Gain Reward (critic disagreement)
        with torch.no_grad():
            q_preds = torch.stack([c(torch.cat([s, z], dim=1)) for c in self.worker_critics])
            info_gain_reward = q_preds.var(dim=0).mean(dim=1)

        intrinsic_reward = self.empowerment_coeff * empowerment_reward + self.infogain_coeff * info_gain_reward
        r += intrinsic_reward.detach()


        # Critic training (similar to T1, but with latent goals)
        td_errors = []
        for i in range(3):
            sg = torch.cat([s, z], dim=1)
            s_next_g = torch.cat([s_next, z], dim=1)
            with torch.no_grad():
                avg_next_q = torch.mean(torch.stack([t(s_next_g) for t in self.worker_targets]), dim=0)
                greedy_a = avg_next_q.argmax(dim=1, keepdim=True)
                min_q_next = torch.min(torch.stack([t(s_next_g).gather(1, greedy_a) for t in self.worker_targets]), dim=0)[0]
                target = r.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1)) * (min_q_next - self.cql_penalty)

            q_values = self.worker_critics[i](sg).gather(1, a_long.unsqueeze(1))
            loss = F.mse_loss(q_values, target)
            self.worker_optimizers[i].zero_grad()
            loss.backward()
            self.worker_optimizers[i].step()
            td_errors.append((q_values - target).abs().detach().numpy())
        self.worker_replay.update_priorities(indices, np.mean(td_errors, axis=0).flatten())
        self._update_targets(self.worker_critics, self.worker_targets)

    def _train_manager(self, batch_size):
        if len(self.manager_replay.buffer) < batch_size: return
        samples, indices = self.manager_replay.sample(batch_size, rer_prob=0.0)
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
                target = r_sum + self.gamma * (1 - done) * q_next
            q_values = self.manager_critics[i](s).gather(1, a_k.unsqueeze(1))
            loss = F.mse_loss(q_values, target.unsqueeze(1))
            self.manager_optimizers[i].zero_grad()
            loss.backward()
            self.manager_optimizers[i].step()
            td_errors.append((q_values - target.unsqueeze(1)).abs().detach().numpy())
        self.manager_replay.update_priorities(indices, np.mean(td_errors, axis=0).flatten())
        self._update_targets(self.manager_critics, self.manager_targets)

    def _update_targets(self, critics, targets):
        for i in range(3):
            for target_param, param in zip(targets[i].parameters(), critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

def run_evaluation(agent, envs):
    total_hits, total_manager_steps = 0, 0
    episodes_to_solve = []

    for env in envs:
        solved = False
        for episode in range(150):
            state = env.reset()
            manager_state, cumulative_r, ep_done = state, 0, False

            while not ep_done:
                manager_action_idx = agent.choose_manager_action(manager_state, None)
                direction_vector = agent.manager_actions[manager_action_idx]

                # Calculate goal state as an offset from the manager's current state
                goal_state = manager_state.copy()
                goal_state[:2] += direction_vector
                goal_state[:2] = np.clip(goal_state[:2], 0, agent.size - 1)

                latent_goal = agent.get_latent_goal(manager_state, goal_state)

                achieved_state = None
                for _ in range(agent.manager_interval):
                    action = agent.choose_action(state, latent_goal)
                    next_state, reward, done, _ = env.step(action)

                    worker_reward = -np.linalg.norm(next_state[:2] - goal_state[:2])
                    agent.worker_replay.add((state, action, worker_reward, next_state, done, latent_goal), done)

                    state = next_state
                    cumulative_r += reward
                    achieved_state = next_state
                    if done:
                        if reward > 5: solved = True
                        ep_done = True; break

                # Manager experience and metrics
                agent.manager_replay.add((manager_state, manager_action_idx, cumulative_r, state, ep_done, goal_state), ep_done)
                if np.linalg.norm(achieved_state[:2] - goal_state[:2]) < 1.5: total_hits += 1
                total_manager_steps += 1

                manager_state, cumulative_r = state, 0
                agent.train()

            if solved:
                episodes_to_solve.append(episode); break

    avg_episodes = np.mean(episodes_to_solve) if episodes_to_solve else float('inf')
    hit_rate = total_hits / total_manager_steps if total_manager_steps > 0 else 0
    return avg_episodes, hit_rate

if __name__ == '__main__':
    envs = [KeyDoorEnv(seed=i) for i in range(4)]
    agent = T3_Agent(state_dim=envs[0]._get_obs().shape[0], action_dim=4, size=envs[0].size)

    avg_episodes, hit_rate = run_evaluation(agent, envs)
    print(f"T3 Agent - Average episodes to first solve: {avg_episodes:.2f}")
    print(f"T3 Agent - Subgoal hit rate: {hit_rate:.2%}")
