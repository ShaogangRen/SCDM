"""Latent-aware SCDM: observed variables plus q explicit latent nodes.

Entry point:  python -m scdm.latent.cli --config configs/latent/<name>.yaml

This package is independent of :mod:`scdm.nonlatent`. Importing one never pulls
in the other, so a run cannot silently mix the two settings.
"""
