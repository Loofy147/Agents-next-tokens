# Proposed Environment Enhancements

This document outlines a series of proposed enhancements to the `KeyDoorEnv` environment, designed to support the agent's new roadmap and provide a richer, more challenging testbed for its growing capabilities.

## Phase 1: Foundational Enhancements

*   **Objective:** Increase the complexity of the environment to provide a more meaningful challenge for the T4 world model.
*   **Action Items:**
    *   **Add Walls and Obstacles:** Introduce walls and obstacles into the grid, requiring the agent to navigate more complex layouts. The procedural generation can be extended to create a variety of maze-like structures.
    *   **Introduce Multiple Rooms:** Expand the environment to include multiple rooms, connected by doors. This will require the agent to learn to navigate a more structured and hierarchical space.
    *   **Add More Object Types:** Introduce new object types beyond keys and doors, such as boxes that can be pushed, switches that affect the environment, and multiple types of keys for different doors.

## Phase 2: Language-Guided Tasks

*   **Objective:** Enable the agent to perform a wider variety of tasks, specified by natural language instructions.
*   **Action Items:**
    *   **Language-Conditioned Goals:** Modify the environment to accept a language-based goal instead of a fixed goal state. For example, "go to the red key" or "open the blue door".
    *   **Introduce Distractor Objects:** Add objects that are not relevant to the current task to test the agent's ability to focus on the specified goal.
    *   **Compositional Tasks:** Create tasks that require a sequence of actions, such as "pick up the key, then open the door, then go to the goal".

## Phase 3: Generative and Open-Ended Environments

*   **Objective:** Move towards a more open-ended and dynamic environment, inspired by the Genie paper and the T7 roadmap.
*   **Action Items:**
    *   **Generative World:** Begin to lay the groundwork for a generative world, where the environment itself can be modified by the agent's actions.
    *   **Curriculum Generation:** Implement a simple curriculum generation system, where the difficulty of the environment is automatically adjusted based on the agent's performance.
    *   **Open-Ended Exploration:** Create a mode where the agent can freely explore the environment without a specific goal, allowing it to discover new skills and knowledge.
