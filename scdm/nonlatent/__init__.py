"""Non-latent SCDM: fully observed variables, shared predictor + shared W.

Entry point:  python -m scdm.nonlatent.cli --config configs/nonlatent/<name>.yaml

This package is independent of :mod:`scdm.latent`. Importing one never pulls in
the other, so a run cannot silently mix the two settings.
"""
