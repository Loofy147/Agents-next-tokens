# Project Documentation: A 10-Year Roadmap for a Hierarchical Reinforcement Learning Agent

This document provides a comprehensive overview of the hierarchical reinforcement learning (HRL) agent, its architecture, the 10-year development roadmap, and instructions for usage and evaluation.

## 1. System Architecture (Current State: T2)

The agent is a sophisticated hierarchical reinforcement learning system designed to solve complex, long-horizon tasks. It is built on a two-level hierarchy and incorporates several state-of-the-art techniques to improve sample efficiency, exploration, and robustness. The entire implementation is self-contained in `hiro_agent.py`.

### Core Components

#### 1.1. HIRO-Style Hierarchy

The agent employs a two-level, HIRO-style (Hierarchical Reinforcement learning with Off-policy correction) architecture:

*   **Manager (High-Level Policy):** The manager operates at a lower temporal resolution. Its responsibility is to set strategic subgoals for the worker to achieve. In the current T2 implementation, the manager's action space consists of selecting a direction vector, which is used to calculate a target state for the worker.
*   **Worker (Low-Level Policy):** The worker operates at every timestep. Its goal is to reach the subgoal set by the manager. The worker receives an intrinsic reward based on its progress toward the current subgoal, which encourages it to follow the manager's directions.

#### 1.2. Goal-Conditioned Inverse Model (T2 Innovation)

To create a rich and effective subgoal representation, the T2 agent uses a **goal-conditioned inverse model**. This model takes the current state (`s`) and a future target state (`s_g`) as input and produces a compact, latent goal embedding (`z`).

*   **Mechanism:** The inverse model is trained to predict the action (`a`) that was taken to get from `s` to `s_g`. The latent goal `z` is an intermediate representation within this network.
*   **Benefit:** This approach allows the manager to set goals in a learned, abstract space, which can be more expressive and easier for the worker to interpret than raw coordinate goals.

#### 1.3. Conservative Q-Critic Ensemble (T1 Innovation)

To combat the overestimation bias common in Q-learning, the agent uses an ensemble of three Q-critics with a conservative update rule:

*   **Ensemble:** Both the manager and the worker maintain three separate Q-networks (critics) and their corresponding target networks.
*   **Conservative Q-Target:** When calculating the target value for a Q-update, the agent uses the minimum Q-value from the ensemble of target networks. A small penalty (CQL-style) is also subtracted to further discourage over-optimism. This leads to more stable and reliable learning.

#### 1.4. Advanced Replay Buffer

The agent's `ReplayBuffer` is a key component for sample efficiency and combines three distinct techniques:

*   **Prioritized Experience Replay (PER):** Transitions with a high TD-error (i.e., transitions where the agent's prediction was very wrong) are more likely to be sampled for training. This focuses the agent's learning on its biggest mistakes.
*   **Hindsight Experience Replay (HER):** The T2 agent uses a latent-goal version of HER. When the worker fails to reach its assigned subgoal, it can retroactively "pretend" that the state it *did* reach was the intended goal. This allows the agent to learn from its failures and improves its ability to generate and achieve latent goals.
*   **Retrieval-Enhanced Replay (RER):** To further improve learning, the replay buffer occasionally retrieves entire past episodes that are semantically similar to the current training batch (measured by cosine similarity of state embeddings) and adds them to the sample. This helps the agent learn from relevant past experiences.

### 2. Environment

The agent is evaluated in a procedurally generated `KeyDoor` environment. This is a simple grid world where the agent must:
1.  Navigate to a key.
2.  Pick up the key.
3.  Navigate to a door.
4.  Open the door to solve the maze.

The procedural generation of the key's position ensures that the agent is evaluated on its ability to generalize rather than memorize a single solution.

## 3. The 10-Year Development Roadmap

The project is structured as a 10-year simulation, where each "year" corresponds to a "token" representing a significant technical leap. Each token builds upon the previous one, incorporating cutting-edge research trends to create a progressively more capable and autonomous agent.

### Seed / Initial Token (T0)

*   **Label:** `HIER-HER-ICM-PER-TRIPLEQ-ADAPT-KEYDOOR`
*   **Meaning:** A foundational hierarchical agent with a HIRO-like manager, Hindsight Experience Replay, an Intrinsic Curiosity Module, Prioritized Experience Replay, Triple-Q critics, and an adaptive epsilon-greedy exploration strategy, all evaluated on the `KeyDoor` benchmark.

---

### Year 1 (T1)

*   **Label:** `HYBRID_QC-HER++ICM-RER`
*   **Leap:** Moves from simple Q-learning to a **conservative ensemble of Q-critics** to reduce overestimation. Introduces **Retrieval-Enhanced Replay (RER)** to learn from semantically similar past experiences. Transitions from numpy-based models to PyTorch.
*   **Implemented:** Yes.

---

### Year 2 (T2)

*   **Label:** `GOAL-INV-HIRO-HER`
*   **Leap:** Introduces a **goal-conditioned inverse model** to generate latent subgoals. The manager now selects reachable future states, which are then encoded into a latent representation for the worker. This makes goal-setting more robust and abstract. Implements **latent goal relabeling** for HER.
*   **Implemented:** Yes.

---

### Year 3 (T3)

*   **Label:** `ICM+EMP-INFOGAIN-AUG`
*   **Leap:** Enhances the agent's intrinsic motivation by combining the ICM with **empowerment** and **information-gain** signals. This will encourage more diverse and intelligent exploration. Adds data augmentation in the representation space for improved robustness.

---

### Year 4 (T4)

*   **Label:** `ENHANCED-HIER-PRIOR-AUX`
*   **Leap:** The manager evolves into a **latent-hierarchical actor** with an auxiliary network that predicts the long-term value of a candidate subgoal. This allows the manager to make more informed, value-driven decisions when selecting subgoals.

---

### Year 5 (T5)

*   **Label:** `ON-PLUG-EXPLORE-META`
*   **Leap:** Introduces **meta-learning** via a lightweight online meta-controller. This controller will tune key hyperparameters—such as the manager's interval, intrinsic reward scaling, and exploration rates—on the fly, adapting the agent's behavior to the specific task distribution.

---

### Year 6 (T6)

*   **Label:** `SEMI-SIM2REAL-ROBUST`
*   **Leap:** Focuses on robustness and generalization by incorporating **domain randomization** and **adversarial perturbations** during training. The agent will be trained on a variety of simulated environments with noisy sensors and randomized dynamics to improve its zero-shot transfer to new, unseen mazes.

---

### Year 7 (T7)

*   **Label:** `GRAPH-RELNET-HIERO`
*   **Leap:** The agent's state representation is upgraded to a **graph-relational encoder**. This will allow the agent to reason about the relationships between entities in the environment (e.g., keys, doors, rooms), and the manager will operate in a latent graph space, setting subgoals at a symbolic/relational level.

---

### Year 8 (T8)

*   **Label:** `OFFPOL-HIRO+TD3-SAC-STYLE`
*   **Leap:** The worker's policy is upgraded from a Q-only, discrete-action policy to a continuous, **actor-critic** policy (a hybrid of TD3 and SAC). This will enable finer-grained control and more stable learning, with full off-policy correction for the hierarchical updates.

---

### Year 9 (T9)

*   **Label:** `LATENT-SYMBOLIC-HIERARCHY`
*   **Leap:** The agent develops the ability for **symbolic latent planning**. It will learn a set of discrete, symbolic representations of its environment and subgoals (e.g., via a VQ-VAE). The manager will then be able to perform model-based rollouts in this symbolic latent space, enabling multi-step planning.

---

### Year 10 (T10)

*   **Label:** `AUTONOMOUS-HIER-SELF-SUPER`
*   **Leap:** The agent becomes an **autonomous lifelong learner**. It will operate continuously, with a self-supervisor that generates its own curriculum, mitigates catastrophic forgetting, and discovers new, reusable skills. The system will be designed to learn, compose, and publish its own compact, symbolic APIs for downstream tasks, representing a significant step toward artificial general intelligence.

## 4. Usage and Evaluation Instructions

This section provides practical instructions for setting up the environment, running the agent, and interpreting the evaluation results.

### 4.1. Environment Setup

The project requires Python 3 and the following libraries:
*   `numpy`
*   `torch`

To create a clean environment and install the dependencies, follow these steps:

1.  **Create a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv hrl_env
    source hrl_env/bin/activate
    ```

2.  **Install the required libraries:**
    ```bash
    pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
    ```
    *Note: We recommend installing the CPU-only version of PyTorch to avoid potential CUDA-related issues and to ensure compatibility with a wider range of machines.*

### 4.2. Running the Agent

The entire agent and its evaluation harness are contained in `hiro_agent.py`. To run the agent, simply execute the script from your terminal:

```bash
python3 hiro_agent.py
```

The script will initialize the T2 agent, run it on a set of 4 procedurally generated `KeyDoor` environments, and print the final evaluation metrics to the console.

### 4.3. Interpreting the Evaluation Metrics

The script will output two key metrics:

*   **Average episodes to first solve:** This metric measures the agent's sample efficiency. It is the average number of episodes the agent required to solve the maze for the first time, averaged across all evaluation environments. A lower number indicates a more sample-efficient agent. An `inf` value means the agent did not solve the maze within the given episode limit.
*   **Subgoal hit rate:** This metric is specific to the T2 agent and measures the effectiveness of the manager's guidance. It is the percentage of subgoals set by the manager that the worker successfully reaches within the given time horizon. A higher hit rate indicates that the manager is setting realistic and achievable subgoals, and that the worker is successfully following the manager's directions.

### 4.4. Log Files

The results of the evaluation runs are stored in log files (e.g., `t2_evaluation.log`). These files provide a record of the agent's performance at each stage of the roadmap.
