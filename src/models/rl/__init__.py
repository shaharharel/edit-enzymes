"""RL optimization loop for enzyme design.

Ties together backbone generation, sequence design, and multi-objective
scoring into an RL training loop with credit-assigned rewards:
- Backbone policy (REINFORCE): geometry + constraint rewards
- Sequence policy (PPO): stability + activity rewards
"""

from src.models.rl.reward import RewardFunction
from src.models.rl.backbone_policy import BackbonePolicy, BackboneValueBaseline
from src.models.rl.sequence_policy import SequencePolicy, SequenceValueHead
from src.models.rl.ppo_trainer import RLTrainer, RolloutBuffer, RolloutEntry, compute_gae
