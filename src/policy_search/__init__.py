"""Policy search utilities for Continuous Authentication.

This package performs an *offline* grid search over lock policies:
  (vote_window_size=N, vote_min_rejects=M, target_window_frr, window_size, overlap)

It reuses trained checkpoints and cached per-window scores, then evaluates
window-level FAR/FRR/ERR and time-to-first-lock stats to output:
  - full grid table
  - Pareto frontier table
  - best_lock_policy.json (optional)
"""

