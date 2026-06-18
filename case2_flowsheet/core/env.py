# -*- coding: utf-8 -*-
"""
Shared environment + seeding constants — byte-identical across both case studies.

NOTE: the per-scenario seed DERIVATION (scenario_seed) differs between the two
cases (case 1 uses the numeric part of the scenario id; case 2 uses a sha256
hash) and is intentionally NOT unified here — see RECONCILIATION.md.
"""
import os

# Shared base seed for both case studies' deterministic per-scenario RNG.
BASE_SEED = 42


def _env_flag(name):
    """Truthy environment flag: anything other than unset/0/false/no."""
    return os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no")
