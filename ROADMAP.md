# Project Roadmap: Towards a Foundation Agent

This document outlines the forward-looking development roadmap for the hierarchical reinforcement learning agent, building upon the T3 architecture. The goal is to evolve the agent from a specialized problem-solver into a generalist, language-guided foundation agent capable of in-context learning and zero-shot generalization.

---

### Year 4 (T4) - Foundational World Model

*   **Label:** `UNIFIED-TRANSFORMER-WM`
*   **Leap:** The agent's architecture will be unified around a single, powerful **Transformer-based world model**. This model, inspired by cutting-edge research like DreamerV3 and IRIS, will learn a comprehensive, predictive model of the environment from raw observations.
*   **Rationale:** A robust world model provides a stable foundation for all future capabilities. It enables the agent to plan entirely in its latent space ("imagination"), drastically improving sample efficiency and forming the basis for more advanced reasoning.

---

### Year 5 (T5) - Causal State Abstraction

*   **Label:** `CAUSAL-DISENTANGLE-WM`
*   **Leap:** The world model will be enhanced with **causal discovery mechanisms**. The agent will learn to build a causal graph of the environment's dynamics, allowing it to distinguish true causal relationships from spurious correlations.
*   **Rationale:** This is a principled approach to generalization. By understanding causality, the agent can make robust predictions about the consequences of its actions, even in novel or out-of-distribution scenarios, moving beyond simple pattern recognition.

---

### Year 6 (T6) - Language-Guided Skill Discovery

*   **Label:** `LANGUAGE-GUIDED-WM-PLAN`
*   **Leap:** The agent will integrate a **language model** to interpret high-level, human-given commands. The manager's role will evolve to translate these natural language instructions into a sequence of subgoals that can be planned and executed by the causal world model.
*   **Rationale:** This leap dramatically expands the agent's usability and task complexity. It allows for flexible, zero-shot instruction following and compositional task execution, bridging the gap between abstract human goals and concrete actions.

---

### Year 7 (T7) - Generative Environment Curriculum

*   **Label:** `GENERATIVE-CURRICULUM-AGENT`
*   **Leap:** The agent will become its own teacher by learning to generate a curriculum of novel tasks. It will leverage a **generative interactive environment** (inspired by papers like "Genie") to create challenges that are perfectly matched to its current capabilities, fostering open-ended skill acquisition.
*   **Rationale:** This step is critical for achieving autonomous learning. By generating its own data and tasks, the agent can overcome the limitations of fixed datasets and environments, exploring and mastering a much wider range of skills.

---

### Year 8 (T8) and Beyond - Foundation Agent Pre-training

*   **Label:** `FOUNDATION-AGENT-PRETRAIN`
*   **Leap:** The agent will be scaled up and pre-trained on a massive and diverse dataset of tasks, many of which will be self-generated. The resulting **foundation agent** will be capable of rapid, in-context adaptation to new, unseen problems, similar to the capabilities demonstrated by Gato and other large-scale agent models.
*   **Rationale:** This is the culmination of the roadmap, producing a single, generalist agent that can be prompted to solve a wide array of tasks without task-specific fine-tuning. This represents a significant step towards artificial general intelligence.
