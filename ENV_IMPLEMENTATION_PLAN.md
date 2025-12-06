# Environment Implementation Plan: Phase 1

This document outlines the step-by-step plan for implementing the Phase 1 enhancements to the `KeyDoorEnv` environment, as defined in `ENVIRONMENT_ENHANCEMENTS.md`.

## 1. Introduce a Grid-Based World

*   **Objective:** Replace the current coordinate-based system with a more flexible and extensible grid-based representation.
*   **Rationale:** A grid-based world will make it easier to add walls, obstacles, and other environmental features.
*   **Action Items:**
    *   In `hiro_agent.py`, modify the `KeyDoorEnv` class to use a 2D NumPy array to represent the environment's grid.
    *   Define integer constants for different grid-cell types (e.g., `EMPTY`, `WALL`, `AGENT`, `KEY`, `DOOR`).
    *   Update the `reset` method to initialize the grid with the agent, key, and door at their respective positions.
    *   Update the `_get_obs` method to return a flattened representation of the grid.

## 2. Add Walls and Obstacles

*   **Objective:** Implement the procedural generation of walls and obstacles.
*   **Rationale:** This will increase the complexity of the navigation task and provide a more challenging environment for the agent.
*   **Action Items:**
    *   In the `reset` method, add a loop to place a random number of walls or obstacles on the grid.
    *   Ensure that there is always a valid path from the agent's starting position to the key and from the key to the door. This can be done using a simple pathfinding algorithm like Breadth-First Search (BFS).
    *   Update the `step` method to prevent the agent from moving into cells that contain walls or obstacles.

## 3. Introduce Multiple Rooms

*   **Objective:** Expand the environment to include multiple rooms connected by doors.
*   **Rationale:** This will introduce a hierarchical structure to the environment, requiring the agent to learn to navigate between rooms.
*   **Action Items:**
    *   Update the procedural generation algorithm to create multiple rooms of different sizes.
    *   Add doors to the walls between rooms.
    *   The agent will need a key to open each door. This will require modifying the `has_key` logic to handle multiple keys and doors.

## 4. Add More Object Types

*   **Objective:** Introduce new object types to increase the variety of tasks the agent can perform.
*   **Rationale:** This will provide a richer set of interactions for the agent and enable the creation of more complex puzzles.
*   **Action Items:**
    *   **Boxes:** Add a new object type for boxes that can be pushed by the agent. This will require updating the `step` method to handle the physics of pushing objects.
    *   **Switches:** Add switches that can be toggled by the agent to affect the environment (e.g., opening and closing doors).
    *   **Multiple Key/Door Colors:** Introduce different colored keys and doors, where each key can only open a door of the same color.
