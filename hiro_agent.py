import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Environment ---
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
            self.has_key = True
            reward = 1.0
        if np.array_equal(self.agent_pos, self.door_pos) and self.has_key:
            reward = 10.0
            done = True
        if self.steps >= self.max_steps:
            done = True
        return self._get_obs(), reward, done, {}

# --- T1 Components ---
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, act_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ICM, self).__init__()
        self.forward_model = nn.Sequential(nn.Linear(state_dim + action_dim, 64), nn.ReLU(), nn.Linear(64, state_dim))

    def forward(self, state, action_one_hot):
        sa = torch.cat([state, action_one_hot], dim=1)
        return self.forward_model(sa)

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
        if len(self.buffer) > self.capacity: self.buffer.pop(0)

        if is_done:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            if len(self.episodes) > 200: self.episodes.pop(0) # Keep last 200 episodes

    def sample(self, batch_size, her_prob=0.8, rer_prob=0.2):
        priorities = np.array(self.priorities)**self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # RER: Retrieval-Enhanced Replay
        if random.random() < rer_prob and len(self.episodes) > 1:
            main_sample_state_embedding = np.mean([s[0] for s in samples], axis=0)
            best_ep_idx = self._get_similar_episode(main_sample_state_embedding)
            retrieved_episode = self.episodes[best_ep_idx]
            extra_samples = random.sample(retrieved_episode, min(len(retrieved_episode), batch_size // 4))
            samples.extend(extra_samples)

        # HER: Hindsight Experience Replay
        processed_samples = []
        for s, a, r, s_next, done, g in samples:
            if random.random() < her_prob:
                # Relabel goal to achieved state
                new_goal = s_next[:2]
                # Recalculate reward based on new goal
                new_reward = 0.0 if np.linalg.norm(s_next[:2] - new_goal) < 1e-5 else -0.1
                processed_samples.append((s, a, new_reward, s_next, done, new_goal))
            else:
                processed_samples.append((s, a, r, s_next, done, g))

        return processed_samples, indices

    def _get_similar_episode(self, target_embedding):
        similarities = []
        for ep in self.episodes:
            ep_embedding = np.mean([s[0] for s in ep], axis=0)
            # Cosine similarity
            sim = np.dot(target_embedding, ep_embedding) / ((np.linalg.norm(target_embedding) * np.linalg.norm(ep_embedding)) + 1e-8)
            similarities.append(sim)
        return np.argmax(similarities)

    def update_priorities(self, indices, new_priorities):
        for i, p in zip(indices, new_priorities):
            self.priorities[i] = p

# --- T1 Hierarchical Agent ---
class T1_HIRO_Agent:
    def __init__(self, state_dim, action_dim, size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.size = size
        self.manager_interval = 10
        self.tau = 0.005
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.cql_penalty = 0.1

        # Worker
        self.worker_critics = [Critic(state_dim + 2, action_dim) for _ in range(3)]
        self.worker_targets = [Critic(state_dim + 2, action_dim) for _ in range(3)]
        self.worker_optimizers = [optim.Adam(c.parameters(), lr=1e-3) for c in self.worker_critics]
        self.worker_replay = ReplayBuffer(10000)

        # Manager (goal space is 10x10 grid positions)
        self.manager_critics = [Critic(state_dim, 100) for _ in range(3)]
        self.manager_targets = [Critic(state_dim, 100) for _ in range(3)]
        self.manager_optimizers = [optim.Adam(c.parameters(), lr=1e-3) for c in self.manager_critics]
        self.manager_replay = ReplayBuffer(1000)

        # ICM
        self.icm = ICM(state_dim, action_dim)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)

    def choose_goal(self, state):
        if random.random() < self.epsilon: return np.random.randint(0, self.size, 2).astype(np.float32)
        state_t = torch.FloatTensor(state)
        with torch.no_grad():
            q_values = torch.mean(torch.stack([c(state_t) for c in self.manager_critics]), dim=0)
        goal_idx = q_values.argmax().item()
        return np.array([goal_idx // 10, goal_idx % 10], dtype=np.float32)

    def choose_action(self, state, goal):
        if random.random() < self.epsilon: return random.randint(0, self.action_dim - 1)
        state_goal = torch.FloatTensor(np.concatenate([state, goal]))
        with torch.no_grad():
            q_values = torch.mean(torch.stack([c(state_goal) for c in self.worker_critics]), dim=0)
        return q_values.argmax().item()

    def train(self, batch_size=64):
        self._train_worker(batch_size)
        self._train_manager(batch_size)
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)

    def _train_worker(self, batch_size):
        if len(self.worker_replay.buffer) < batch_size: return
        samples, indices = self.worker_replay.sample(batch_size)
        s, a, r, s_next, done, g = map(np.array, zip(*samples))

        s = torch.FloatTensor(s)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        s_next = torch.FloatTensor(s_next)
        done = torch.FloatTensor(done)
        g = torch.FloatTensor(g)

        # ICM Training
        a_one_hot = F.one_hot(a, num_classes=self.action_dim).float()
        pred_s_next = self.icm(s, a_one_hot)
        icm_loss = F.mse_loss(pred_s_next, s_next)
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

        # Critic Training
        td_errors = []
        for i in range(3):
            sg = torch.cat([s, g], dim=1)
            s_next_g = torch.cat([s_next, g], dim=1)

            with torch.no_grad():
                avg_next_q = torch.mean(torch.stack([t(s_next_g) for t in self.worker_targets]), dim=0)
                greedy_a = avg_next_q.argmax(dim=1, keepdim=True)

                q_next_vals = torch.stack([self.worker_targets[j](s_next_g).gather(1, greedy_a) for j in range(3) if j != i])
                min_q_next = torch.min(q_next_vals, dim=0)[0]
                target = r.unsqueeze(1) + self.gamma * (1 - done.unsqueeze(1)) * (min_q_next - self.cql_penalty)

            q_values = self.worker_critics[i](sg).gather(1, a.unsqueeze(1))
            loss = F.mse_loss(q_values, target)

            self.worker_optimizers[i].zero_grad()
            loss.backward()
            self.worker_optimizers[i].step()
            td_errors.append((q_values - target).abs().detach().numpy())

        self.worker_replay.update_priorities(indices, np.mean(td_errors, axis=0).flatten())
        self._update_targets(self.worker_critics, self.worker_targets)

    def _train_manager(self, batch_size):
        if len(self.manager_replay.buffer) < batch_size: return
        # Manager samples are (state, goal_idx, cumulative_r, next_state, done, goal_pos)
        samples, indices = self.manager_replay.sample(batch_size, her_prob=0.0) # No HER for manager
        s, g_idx, r_sum, s_next, done, _ = map(np.array, zip(*samples))

        s = torch.FloatTensor(s)
        g_idx = torch.LongTensor(g_idx.astype(int))
        r_sum = torch.FloatTensor(r_sum)
        s_next = torch.FloatTensor(s_next)
        done = torch.FloatTensor(done)

        td_errors = []
        for i in range(3):
            with torch.no_grad():
                q_next = torch.mean(torch.stack([t(s_next) for t in self.manager_targets]), dim=0).max(1)[0]
                target = r_sum + self.gamma * (1 - done) * q_next

            q_values = self.manager_critics[i](s).gather(1, g_idx.unsqueeze(1))
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

    def get_intrinsic_reward(self, state, action, next_state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
            action_t = F.one_hot(torch.tensor(action), num_classes=self.action_dim).float().unsqueeze(0)
            pred_next_state = self.icm(state_t, action_t)
            return F.mse_loss(pred_next_state, next_state_t).item()

def run_evaluation(agent, envs):
    episodes_to_solve = []
    for env in envs:
        solved = False
        for episode in range(250):
            state = env.reset()
            manager_state, cumulative_r, ep_done = state, 0, False

            while not ep_done:
                goal = agent.choose_goal(manager_state)
                for _ in range(agent.manager_interval):
                    action = agent.choose_action(state, goal[:2])
                    next_state, reward, done, _ = env.step(action)

                    intrinsic_reward = agent.get_intrinsic_reward(state, action, next_state)
                    # Scale and clip the intrinsic reward
                    scaled_reward = np.clip(0.1 * intrinsic_reward, 0, 1)
                    agent.worker_replay.add((state, action, reward + scaled_reward, next_state, done, goal[:2]), done)

                    state = next_state
                    cumulative_r += reward
                    if done:
                        if reward > 5: solved = True
                        ep_done = True
                        break

                goal_idx = int(goal[0] * 10 + goal[1])
                agent.manager_replay.add((manager_state, goal_idx, cumulative_r, state, ep_done, goal), ep_done)
                manager_state, cumulative_r = state, 0
                agent.train()

            if solved:
                episodes_to_solve.append(episode)
                break
    return np.mean(episodes_to_solve) if episodes_to_solve else float('inf')


if __name__ == '__main__':
    envs = [KeyDoorEnv(seed=i) for i in range(16)]
    agent = T1_HIRO_Agent(state_dim=envs[0]._get_obs().shape[0], action_dim=4, size=envs[0].size)

    avg_episodes = run_evaluation(agent, envs)
    print(f"T1 Agent (Corrected) - Average episodes to first solve: {avg_episodes:.2f}")
