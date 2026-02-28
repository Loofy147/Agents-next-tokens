import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class Skill:
    name: str
    description: str
    q_score: float
    dimensions: Dict[str, float]
    metadata: Dict[str, Any]

class SkillRegistry:
    """
    A modular registry for agent capabilities ("skills").
    Inspired by the High-Q Skill Registry.
    """
    def __init__(self, registry_path: str = "SKILL_REGISTRY_JSON.json"):
        self.registry_path = registry_path
        self.skills: Dict[str, Skill] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                for name, s_data in data.items():
                    self.skills[name] = Skill(**s_data)

    def save(self):
        data = {name: asdict(skill) for name, skill in self.skills.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def register_skill(self, skill: Skill):
        self.skills[skill.name] = skill
        self.save()

    def get_skill(self, name: str) -> Optional[Skill]:
        return self.skills.get(name)

    def list_skills(self) -> List[str]:
        return list(self.skills.keys())

    def generate_markdown_registry(self) -> str:
        lines = ["# Agent Skill Registry", "", "| Skill Name | Q-Score | Description |", "|------------|---------|-------------|"]
        for name, skill in sorted(self.skills.items(), key=lambda x: x[1].q_score, reverse=True):
            lines.append(f"| {name} | {skill.q_score:.4f} | {skill.description} |")
        return "\n".join(lines)
