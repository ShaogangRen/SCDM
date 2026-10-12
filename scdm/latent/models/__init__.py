"""Latent-aware SCDM model components."""

from .graph_head import LatentBlockGraphHead, build_latent_graph_head
from .predictor import LatentCausalModel

__all__ = ["LatentCausalModel", "LatentBlockGraphHead", "build_latent_graph_head"]
