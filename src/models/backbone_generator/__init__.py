"""Backbone generator models.

Two options:
- SE3BackboneDiffusion: Custom EGNN/IPA model (trained from scratch, research use)
- RFdiffusionWrapper: Pretrained RFdiffusion (production use, requires external install)
"""

from src.models.backbone_generator.base import AbstractBackboneGenerator
from src.models.backbone_generator.diffusion_model import SE3BackboneDiffusion, DiffusionConfig
from src.models.backbone_generator.noise_schedule import DiffusionSchedule
from src.models.backbone_generator.rfdiffusion_wrapper import RFdiffusionWrapper, RFdiffusionConfig
