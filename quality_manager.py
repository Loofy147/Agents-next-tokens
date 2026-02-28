import numpy as np
from typing import Dict, List, Any
import datetime

class QualityManager:
    """
    Manages the "High-Q" evaluation framework.
    Evaluates system components across six dimensions:
    - G (Grounding): Empirical validity and factual accuracy.
    - C (Certainty): Confidence levels and probabilistic calibration.
    - S (Structure): Logical hierarchy and organization.
    - A (Applicability): Direct utility and actionability.
    - H (Coherence): Internal consistency and synergy.
    - V (Generativity): Potential for transfer learning and emergent patterns.
    """
    def __init__(self):
        self.weights = {
            'G': 0.18, 'C': 0.20, 'S': 0.18, 'A': 0.16,
            'H': 0.12, 'V': 0.16
        }
        self.history = []

    def calculate_q_score(self, dimensions: Dict[str, float]) -> float:
        """
        Calculates the weighted Q-Score from dimension scores (0.0 to 1.0).
        """
        score = sum(self.weights.get(d, 0) * dimensions.get(d, 0) for d in self.weights)
        # Normalize in case weights don't sum to 1 (they do here, but for robustness)
        total_weight = sum(self.weights.get(d, 0) for d in dimensions if d in self.weights)
        if total_weight > 0:
            score /= total_weight
        return score

    def evaluate_component(self, name: str, dimensions: Dict[str, float]) -> float:
        q_score = self.calculate_q_score(dimensions)
        entry = {
            'component': name,
            'q_score': q_score,
            'dimensions': dimensions,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.history.append(entry)
        return q_score

    def get_summary(self) -> str:
        if not self.history:
            return "No evaluations recorded."

        avg_q = np.mean([h['q_score'] for h in self.history])
        return f"Average System Q-Score: {avg_q:.4f} across {len(self.history)} components."
