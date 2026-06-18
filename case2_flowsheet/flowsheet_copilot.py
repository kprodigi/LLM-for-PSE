# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  FlowsheetCopilotR1 — Case Study 2                                             ║
║  Admissibility-aware, fail-closed, tool-augmented LLM co-pilot for             ║
║  Aspen Plus distillation flowsheet optimization.                               ║
║                                                                                ║
║  The LLM PLANS and EXPLAINS; it does NOT compute the solution. A numerical     ║
║  optimizer + Aspen Plus (no surrogate option) decide and verify the numbers.   ║
║  Retrieval grounds the plan in versioned process knowledge. A recommendation   ║
║  is released ONLY if a tool-verified admissibility certificate passes; on      ║
║  failure the system fails closed (abstains or returns a verified minimal-      ║
║  deviation safe alternative).                                                  ║
║                                                                                ║
║  Conventions/kernel mirror case study 1 (AxiomCodeR3.py): RAGEngine,           ║
║  MCP-style in-process tool registry, AdmissibilityCertificate + passes(),      ║
║  explicit per-scenario numpy Generators from BASE_SEED, mock modes + scenario  ║
║  subset via env vars, export_results + manifest + diagnostics + figures.       ║
║                                                                                ║
║  SCOPE: feasibility is guaranteed RELATIVE TO THE REAL ASPEN PLUS V14 MODEL    ║
║  and the encoded constraints -- NOT relative to a physical plant.              ║
║                                                                                ║
║  Env vars (prefix FCO_):                                                       ║
║    FCO_MOCK_ASPEN=1   the surrogate was REMOVED; real Aspen Plus V14 COM only  ║
║    FCO_MOCK_LLM=1     use a fixed decision spec instead of calling Anthropic   ║
║    FCO_SCENARIOS=...  comma-separated case-id subset for fast iteration        ║
║    ANTHROPIC_API_KEY  read lazily; never hardcoded/printed/logged              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import os
import sys
import re
import json
import time
import math
import logging
import hashlib
import platform
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np


# Vendored kernel: this case study carries its OWN copy of the kernel at
# case2_flowsheet/core/ (self-contained; no shared top-level core). Repo root on
# sys.path so `case2_flowsheet.core` resolves when run as a script, via -m, or under
# tests; the two case studies' kernels are independent copies by design.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case2_flowsheet.core.env import _env_flag, BASE_SEED
from case2_flowsheet.core.tools import MCPToolRegistry
from case2_flowsheet.core.diagnostics import make_diagnostics_logger


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG / ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

# Anthropic key is read from the environment and used lazily (see
# FlowsheetCopilotAgent._client_or_raise): a missing key only errors when a live
# LLM call is actually attempted, never at import and never in mock mode.
API_KEY   = os.environ.get("ANTHROPIC_API_KEY")
# Primary planner model (env-overridable). The cross-model arm (C4: the guarantee is
# independent of model quality) runs a second, weaker/cheaper Claude — chosen at
# live-run time via FCO_LLM_MODEL_WEAK; in the $0 dry-run it is only a label on the
# fake response. Per-call model is recorded in the capture, so a mixed-model run is
# fully auditable.
LLM_MODEL      = os.environ.get("FCO_LLM_MODEL", "claude-sonnet-4-6")
LLM_MODEL_WEAK = os.environ.get("FCO_LLM_MODEL_WEAK", "claude-haiku-4-5")

# Live-LLM (R6) pricing + caps for the cost PROJECTION/probe — same model and prices
# as case study 1 (claude-sonnet-4-6). USD per token. Update if the model/price moves.
LLM_PRICE_IN_PER_TOK  = 3e-6     # $3 / MTok input
LLM_PRICE_OUT_PER_TOK = 15e-6    # $15 / MTok output
# A full 3-spec decision serializes to ~1200 tokens; a live LLM adds prose, so the old
# 1500 cap risked TRUNCATION (CS1's bug: a truncated spec parsed as garbage). 3072 gives
# ~2.5x headroom AND _call_llm detects stop_reason=='max_tokens' so a truncated response
# is NEVER parsed. Worst-case cost stays well under a $2-3 ceiling.
# Raised 3072 -> 4096 after the live run: the REAL sonnet proposer emits ~2350-2870 output
# tokens (3 verbose candidate specs), i.e. ~93% of 3072 -> too close to the truncation
# boundary for a one-shot run. 4096 gives headroom; _call_llm still detects stop_reason==
# 'max_tokens' and refuses to parse a truncated response. Worst-case cost stays < the cap.
LLM_MAX_TOKENS        = 4096     # per-call output cap (also the worst-case out tokens)
# Planning is a COMPILE task (query -> machine-checkable spec), so temperature 0 is the
# default (deterministic, reproducible). Captured per call. Override via FCO_TEMP.
LLM_TEMP              = float(os.environ.get("FCO_TEMP", "0.0") or 0.0)
# HARD spend backstop for a live run: once the metered cost crosses this, the next live
# call fails closed (never silently overspends). DEFAULTS TO $5 (fail-safe: a cap is
# always armed even if FCO_COST_ABORT_USD is forgotten); override via FCO_COST_ABORT_USD
# (e.g. 2-3). A live run REFUSES to start if this is <=0. The cost PROBE makes no call.
LLM_COST_ABORT_USD    = float(os.environ.get("FCO_COST_ABORT_USD", "5.0") or 5.0)
# Contamination abort (live): if the fraction of live plans yielding NO usable spec
# (unparseable, or no schema-valid spec) exceeds this once enough plans have run, STOP
# the paid run — garbage in, stop paying. DISTINCT from the gate rejecting a well-formed
# spec, which is a reported finding (fail-closed), not contamination.
LLM_CONTAMINATION_ABORT = 0.5
LLM_CONTAMINATION_MIN_N = 4


class LLMTruncatedError(RuntimeError):
    """Raised when a live response hit max_tokens (truncated): non-retryable and not
    parseable — fail loudly so the cap is raised, never parse the garbage (CS1 bug)."""


def _estimate_tokens(text):
    """Offline token estimate (~4 chars/token) for the R6 cost PROJECTION — makes NO
    API call. Deliberately rough; an approved live run meters exact usage from the
    response (FlowsheetCopilotAgent.llm_cost_usd)."""
    return max(1, int(len(str(text)) / 4))


MOCK_ASPEN = _env_flag("FCO_MOCK_ASPEN")
MOCK_LLM   = _env_flag("FCO_MOCK_LLM")
# BASE_SEED imported from case2_flowsheet.core.env (vendored value, 42)

# Results live next to this module (case2_flowsheet/results/), independent of the
# working directory (repo reorg; was "./flowsheet_copilot_results" at the old root).
# Env-overridable (FCO_RESULT_DIR) so the comprehensive campaign / dry-run writes to a
# SEPARATE dir and never overwrites the locked study (live_results/). Set the env BEFORE
# importing this module (capture + harness both read this module-global).
RESULT_DIR = (os.environ.get("FCO_RESULT_DIR") or
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(RESULT_DIR, exist_ok=True)

# Diagnostics log: solver/Aspen failures are recorded here, never swallowed.
diag_log = make_diagnostics_logger("fco.diagnostics",
                                   os.path.join(RESULT_DIR, "diagnostics.log"),
                                   logging.INFO)


def scenario_seed(sid, base=BASE_SEED):
    """Deterministic per-scenario seed from the base seed + the scenario id, so a
    case reproduces identically whether run alone or inside the full sweep."""
    h = int(hashlib.sha256(str(sid).encode()).hexdigest()[:8], 16)
    return (base * 100003 + h) % (2**31 - 1)


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM / DECISION / CONSTRAINT / THRESHOLD CONFIG  (the Phase-0 centerpiece)
# ══════════════════════════════════════════════════════════════════════════════
#
# All thresholds are AUTHOR-DECISION values, recorded in run_manifest.json and
# documented in BUILD_NOTES.md. Constraint thresholds are cited from the
# versioned corpus (source_id / source_version) built in Phase 3.

CONFIG = {
    "system": {
        # MVP separation (decided): methanol/water in one RadFrac, NRTL.
        # Fallback if NRTL convergence is troublesome during real-Aspen bring-up:
        # benzene/toluene under Peng-Robinson (near-ideal, very stable). The
        # backend actually used is recorded at runtime in the manifest.
        "name":            "methanol_water",
        "components":      ["METHANOL", "WATER"],
        "property_method": "NRTL",
        "fallback": {
            "name": "benzene_toluene", "components": ["BENZENE", "TOLUENE"],
            "property_method": "PENG-ROB",
        },
        "feed": {            # saturated-liquid feed, ~1 atm, ~100 kmol/h, z~0.5
            "flow_kmol_h":   100.0,
            "z_methanol":    0.50,     # nominal; drift scenarios vary this
            "vapor_fraction": 0.0,     # saturated liquid
            "pressure_bar":  1.013,
        },
        "column": {
            "n_stages":      25,       # incl. condenser(1) + reboiler(N)
            "feed_stage":    13,       # nominal mid-column feed
            "condenser":     "total",
            "reboiler":      "kettle",
            "pressure_bar":  1.013,    # fixed for the MVP (see decision_vars)
        },
        "baseline_bkp": os.path.join(RESULT_DIR, "baseline_meoh_water.bkp"),
        # Real-Aspen hydraulics/utility AUTHOR DECISIONs (R0/R2; recorded in
        # the manifest). flooding_approach on the real path is a fixed-cross-
        # section vapor-throughput proxy = max stage vapor / v_flood_ref, read
        # from the real Aspen stage vapor profile (\...\COL\Output\VAP_FLOW).
        # It is NOT Aspen's rigorous tray-rating %flood (column internals are not
        # reliably authored headlessly); flooding is slack at the min-duty
        # optimum, so this does not affect the headline metrics. See BUILD_NOTES.
        "hydraulics": {
            "v_flood_ref_kmol_h": 320.0,  # flood vapor capacity (baseline V=175 -> ~0.55)
            "cool_util_C": 30.0,          # condenser coolant temp (cond ΔT approach)
            "heat_util_C": 125.0,         # reboiler steam temp (reb ΔT approach)
        },
        # Model-fidelity perturbation (D3, FCO_TRAY_EFFICIENCY). eta=1.0 is the
        # equilibrium-stage baseline the real RadFrac already runs, so the normal real
        # path NEVER touches this node. For a degraded-fidelity sweep (eta<1) on the
        # REAL path we set a uniform RadFrac Murphree VAPOR efficiency; the tree node
        # below is PENDING R0 verification (set it on the next live build, as with the
        # other empirically-confirmed paths). Until configured, _apply_tray_efficiency
        # RAISES for eta<1 rather than silently running at equilibrium (which would make
        # the fidelity sweep falsely report zero model-relative breaches).
        "fidelity": {
            "murphree_vap_eff_path": None,   # e.g. r"\Data\Blocks\COL\Input\EFF_VAP" — R0-verify first
        },
    },

    # Decision variables consumed by the optimizer and audited by the gate. The
    # integer feed stage is handled by rounding INSIDE evaluate_fn (documented).
    # Operating pressure is supported but OFF by default in the MVP (optimize
    # False) to keep dimensionality/convergence tame; flip optimize=True to add.
    "decision_vars": [
        {"name": "reflux_ratio", "type": "float", "lower": 0.5, "upper": 8.0,
         "optimize": True,  "aspen_path": r"\Data\Blocks\COL\Input\BASIS_RR"},
        {"name": "DtoF",         "type": "float", "lower": 0.42, "upper": 0.58,
         "optimize": True,  "aspen_path": r"\Data\Blocks\COL\Input\BASIS_D"},
        # D:F bounds narrowed in R3 from the earlier [0.30,0.70]: mass balance
        # pins the cut to the feed methanol fraction (D:F~=z), so for z in the
        # nominal+drift range [0.45,0.55] the feasible cut lives in ~[0.45,0.55];
        # [0.42,0.58] covers it with robustness-band margin and lets the optimizer
        # resolve the thin feasible strip instead of wasting evals on D:F<<z (which
        # dumps methanol to the bottoms, violating the 0.01 spec). Rationale logged.
        {"name": "feed_stage",   "type": "int",   "lower": 5,   "upper": 20,
         "optimize": True,  "aspen_path": r"\Data\Blocks\COL\Input\FEED_STAGE\FEED"},
        {"name": "pressure_bar", "type": "float", "lower": 1.0, "upper": 1.5,
         "optimize": False, "aspen_path": r"\Data\Blocks\COL\Input\PRES1"},
    ],

    "objective": {"name": "reboiler_duty", "sense": "min", "units": "kW"},

    # Constraints. relation in {">=","<="}; scale normalizes the signed margin so
    # one buffer applies across heterogeneous units. critical => decision-critical
    # (must be cited; counts toward evidence_coverage).
    "constraints": [
        {"name": "dist_methanol_purity", "relation": ">=", "threshold": 0.99,
         "scale": 0.01, "units": "molefrac", "critical": True,
         "source_id": "SPEC-DIST-PURITY", "source_version": "2.0"},
        {"name": "bottoms_methanol",     "relation": "<=", "threshold": 0.01,
         "scale": 0.01, "units": "molefrac", "critical": True,
         "source_id": "SPEC-BOT-PURITY",  "source_version": "2.0"},
        {"name": "flooding_approach",    "relation": "<=", "threshold": 0.80,
         "scale": 0.10, "units": "frac", "critical": True,
         "source_id": "LIM-FLOODING",    "source_version": "1.0"},
        {"name": "reboiler_duty",        "relation": "<=", "threshold": 5000.0,
         "scale": 1000.0, "units": "kW", "critical": True,
         "source_id": "LIM-REB-DUTY",    "source_version": "1.0"},
        {"name": "cond_temp_approach",   "relation": ">=", "threshold": 5.0,
         "scale": 5.0, "units": "K", "critical": False,
         "source_id": "LIM-DT-MIN",      "source_version": "1.0"},
        {"name": "reb_temp_approach",    "relation": ">=", "threshold": 5.0,
         "scale": 5.0, "units": "K", "critical": False,
         "source_id": "LIM-DT-MIN",      "source_version": "1.0"},
    ],

    # Fail-closed gate thresholds (AUTHOR DECISIONs; recorded in the manifest).
    # Finalized in R2 from the REAL-Aspen characterization (see BUILD_NOTES R2).
    "gate": {
        "buffer":  0.2,    # min normalized constraint margin must be >= buffer.
                           # 0.2 ~= 0.2 mol% purity band above spec. R2: the real
                           # feasible region supports a positive buffer at the
                           # +/-0.3% band (robust optimum has feas_frac 1.0); a
                           # wider +/-0.5% band would forbid any positive buffer.
        "r_min":   0.95,   # >= 95% of the feed-drift ensemble must hold (unified with
                           # case1_reactor ROBUSTNESS_THRESHOLD=0.95; robust optimum
                           # holds at feas_frac 1.0 so the released design is unaffected)
        "e_min":   1.0,    # all decision-critical constraints must be cited
        "rec_rel_tol": 1e-3,   # decision_equals_verified relative tol on duty
        "rec_abs_tol": {"reflux_ratio": 0.10, "DtoF": 0.01, "feed_stage": 0.5,
                        "pressure_bar": 0.02},
        "rec_safe_margin": 0.0,  # LLM-proposed point must clear all margins by this
    },

    # Robustness ensemble used inside the gate: re-evaluate the CHOSEN (fixed)
    # design across a LOCAL feed-composition band z_nom +/- delta.
    # R2 (EMPIRICAL, real Aspen): the 0.99/0.01 DUAL spec mass-balance-PINS the cut
    # to the feed methanol fraction (D:F ~= z), so a FIXED setpoint tolerates only
    # ~+/-0.4-0.5% feed drift on bottoms/purity REGARDLESS of reflux. The
    # prescribed +/-2% band is therefore physically impossible (it would make every
    # design non-robust -> degenerate all-abstain). +/-0.3% sits just inside the
    # mass-balance ceiling: robustness is meaningful (off-strip / sub-min-reflux
    # designs fail; the min-duty design holds), AND a positive purity buffer is
    # still feasible. Documented deviation from the +/-2% default, per the task.
    "robustness": {
        "param": "z_methanol",
        "delta": 0.003,          # +/- 0.3% absolute feed-methanol band (R2)
        # MONTE-CARLO ensemble size (default 64). Env-tunable because, with the surrogate
        # removed, each sample is now a real Aspen COM solve nested inside the optimizer
        # loop; a smaller budget keeps real-Aspen runs tractable. Production default unchanged.
        "n_samples": int(os.environ.get("FCO_ROBUST_N_SAMPLES", "64")),
        "seed": 0,               # fixed seed -> deterministic/reproducible MC sample
                                 # (preserves plan()'s no-perturbation guarantee)
    },

    "optimizer": {
        "backend_mock":  "differential_evolution",  # cheap backend for synthetic test fns
        "backend_real":  "gp_bo",                    # sample-efficient for Aspen
        "de_maxiter":    int(os.environ.get("FCO_DE_MAXITER", "40")), "de_popsize": 12, "de_tol": 1e-4,
        # GP-BO budget (defaults 8 init + 24 iters). Env-tunable for tractable real-Aspen runs.
        "bo_init": int(os.environ.get("FCO_BO_INIT", "8")),
        "bo_iters": int(os.environ.get("FCO_BO_ITERS", "24")),
        "consensus_K":   int(os.environ.get("FCO_CONSENSUS_K", "3")),  # verifier consensus: K candidate specs
        "run_timeout_s": int(os.environ.get("FCO_RUN_TIMEOUT_S", "120")),  # per real-Aspen run watchdog
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MCP-STYLE (IN-PROCESS) TOOL REGISTRY  (reused from case study 1)
# ══════════════════════════════════════════════════════════════════════════════

# MCPToolRegistry now lives in core/tools.py (behaviour-identical; case study 2
# uses the default args_maxlen=120). Imported at the top of this module.


# ══════════════════════════════════════════════════════════════════════════════
# ADMISSIBILITY CERTIFICATE + FAIL-CLOSED GATE  (adapted from case study 1)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AdmissibilityCertificate:
    """
    Deterministic, fail-closed gate over a candidate flowsheet design. A
    recommendation is released ONLY if passes() holds; otherwise the system
    abstains or returns a verified minimal-deviation safe alternative.
    """
    converged: bool                  # Aspen converged AND optimizer terminated cleanly
    margins_ok: bool                 # min normalized constraint margin >= buffer
    robust: bool                     # robustness (feasible fraction) >= r_min
    evidence_ok: bool                # evidence coverage >= e_min AND every explainer
                                     #   citation is a RETRIEVED source (no fabrication)
    recommendation_safe: bool        # LLM-proposed point, simulated, clears all margins
    decision_equals_verified: bool   # released decision == verified optimizer result
    # The 6 checks above are the UNIVERSAL admissibility family (shared with CS1; the
    # consistency slot is decision_equals_verified here, explanation_consistent in CS1).
    # The explainer split adds explanation_consistent below so CS2 measures the SAME five
    # hallucination modes (HARMONIZATION_SPEC §2/§4). It defaults True so a certificate
    # built WITHOUT an explainer (B2, the per-candidate consensus certs) is unaffected.
    explanation_consistent: bool = True   # the post-hoc explainer's stated action ==
                                          # the released decision (no narrative drift)
    margins: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)   # min_margin, robustness, coverage, ...
    reason: str = ""


def passes(cert):
    """A certificate is admissible iff EVERY check holds (fail-closed)."""
    return bool(cert.converged and cert.margins_ok and cert.robust and
                cert.evidence_ok and cert.recommendation_safe and
                cert.decision_equals_verified and cert.explanation_consistent)


def constraint_margins(outputs, constraints):
    """
    Signed, NORMALIZED constraint margins from a (converged) Aspen Plus
    result. For '>=': margin=(value-threshold)/scale; for '<=':
    margin=(threshold-value)/scale. Positive => feasible. A single dimensionless
    buffer therefore applies across heterogeneous units (purity, kW, K, frac).
    Returns {margins:{name:m}, min_margin, feasible, violated:[names]}.
    """
    margins, violated = {}, []
    if not outputs.get("converged"):
        return {"margins": {}, "min_margin": -9.99, "feasible": False,
                "violated": ["__non_converged__"]}
    for c in constraints:
        val = outputs.get(c["name"])
        if val is None:
            margins[c["name"]] = -9.99; violated.append(c["name"]); continue
        raw = (val - c["threshold"]) if c["relation"] == ">=" else (c["threshold"] - val)
        m = raw / c["scale"]
        margins[c["name"]] = float(m)
        if m < 0:
            violated.append(c["name"])
    mn = min(margins.values()) if margins else -9.99
    return {"margins": margins, "min_margin": float(mn),
            "feasible": (mn >= 0.0 and not violated), "violated": violated}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE STUBS  (fleshed out in later phases; Phase 0 keeps the pipeline running)
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_KEYS = ("reboiler_duty", "dist_methanol_purity", "bottoms_methanol",
               "flooding_approach", "cond_temp_approach", "reb_temp_approach")


class AspenFlowsheet:
    """
    The column model: the real Aspen Plus V14 COM backend ONLY (the analytical
    surrogate has been removed -- column design is never approximated). It returns
    a result schema: {converged, <OUTPUT_KEYS>, error, extras...}. Every evaluation is
    appended to self.trace (inputs, outputs, convergence) — the audit artifact.

    Non-negotiables honored: non-converged runs return converged=False with NO
    objective value (never a penalty); the evaluation draws NO
    random numbers (so robustness/verification sampling cannot perturb a
    scenario's RNG stream); every COM call is success/convergence-checked and a
    per-run watchdog kills/hangs are handled; no bare excepts.

    SCOPE: outputs are model-relative (real Aspen Plus V14), not a plant.
    """

    def __init__(self, config=CONFIG, mock=False, rng=None):
        if mock or MOCK_ASPEN:
            raise RuntimeError(
                "The analytical surrogate has been removed: column design requires the real "
                "Aspen Plus V14 COM backend. mock=True and FCO_MOCK_ASPEN are no longer "
                "supported; run with Aspen Plus available.")
        self.config  = config
        self.mock    = False
        self.rng     = rng if rng is not None else np.random.default_rng(BASE_SEED)
        self.trace   = []
        self.F       = config["system"]["feed"]["flow_kmol_h"]
        self.z_nom   = config["system"]["feed"]["z_methanol"]
        self.backend = "aspen-com(V14)"
        self._app    = None       # lazy COM document
        self._opened = None

    # ── unified evaluation ────────────────────────────────────────────────────
    def evaluate(self, decision, feed_z=None):
        z = self.z_nom if feed_z is None else float(feed_z)
        res = self._aspen_eval(decision, z)   # real Aspen Plus only; surrogate removed
        self.trace.append({"decision": {k: (round(float(v), 4)
                                            if isinstance(v, (int, float, np.floating)) else v)
                                        for k, v in decision.items()},
                           "feed_z": round(z, 4), "converged": res["converged"],
                           "objective": res.get("reboiler_duty"),
                           "error": res.get("error")})
        return res

    # ── analytical surrogate: REMOVED ─────────────────────────────────
    def _surrogate(self, decision, z):
        """REMOVED: the analytical surrogate is no longer available. Column design is
        verified only on the real Aspen Plus V14 COM backend (see _aspen_eval)."""
        raise RuntimeError(
            "The analytical surrogate has been removed; column design uses real Aspen "
            "Plus only (mock=True / FCO_MOCK_ASPEN are unsupported).")

    # ── real Aspen Plus COM backend ───────────────────────────────────────────
    # Node paths below are empirically CONFIRMED against the real Aspen Plus V14
    # tree (R0: probe + Variable Explorer), not assumed. resolve_node() still logs
    # any miss and _set_decision/_set_feed_z RAISE (never silently skip) on one.
    # REB_DUTY is
    # Watt (SI baseline) -> kW in _read_outputs. Flooding is a vapor-throughput
    # proxy from the real stage vapor profile (see _flooding), not a tree node.
    _OUT_PATHS = {
        "reboiler_duty":        r"\Data\Blocks\COL\Output\REB_DUTY",        # Watt
        "condenser_duty":       r"\Data\Blocks\COL\Output\COND_DUTY",       # Watt (richer output)
        "dist_methanol_purity": r"\Data\Streams\DIST\Output\MOLEFRAC\MIXED\METHANOL",
        "bottoms_methanol":     r"\Data\Streams\BOT\Output\MOLEFRAC\MIXED\METHANOL",
        "top_temp":             r"\Data\Blocks\COL\Output\TOP_TEMP",        # degC
        "bottom_temp":          r"\Data\Blocks\COL\Output\BOTTOM_TEMP",     # degC
        "vap_profile":          r"\Data\Blocks\COL\Output\VAP_FLOW",        # kmol/h per stage
        "blkstat":              r"\Data\Blocks\COL\Output\BLKSTAT",         # 0 = converged
    }

    def _ensure_com(self):
        """Early-bind via gencache (REQUIRED, R0-verified): late binding cannot
        invoke Engine.Run2 on this multi-interface object (raises AttributeError)."""
        if self._app is not None:
            return self._app
        from win32com.client import gencache              # pywin32
        try:
            self._app = gencache.EnsureDispatch("Apwn.Document")
        except Exception as err:
            raise RuntimeError(
                "Aspen COM early binding (gencache.EnsureDispatch) failed; the real "
                "path requires it because late binding cannot call Engine.Run2: "
                f"{type(err).__name__}: {err}")
        try:
            self._app.SuppressDialogs = 1
        except Exception as err:
            diag_log.warning("Aspen SuppressDialogs failed: %s", err)
        return self._app

    def open(self, case_path=None):
        case_path = case_path or self.config["system"]["baseline_bkp"]
        if not os.path.exists(case_path):
            raise FileNotFoundError(
                f"Baseline Aspen file not found: {case_path}. Build it "
                f"reproducibly with `python build_baseline.py` (real Aspen Plus "
                f"V14), or manually per ASPEN_BASELINE_CHECKLIST.md.")
        app = self._ensure_com()
        app.InitFromArchive2(os.path.abspath(case_path))  # open ONCE (a 2nd call raises)
        try:
            app.Visible = False
        except Exception as err:
            diag_log.warning("Aspen Visible=False failed: %s", err)
        self._opened = case_path
        return app

    def _ensure_open(self):
        """Open the baseline once; re-dispatch+reopen if a watchdog killed it."""
        if self._app is None or self._opened is None:
            self.open()
        return self._app

    def reset_to_baseline(self):
        """Independence WITHOUT reloading the file (a 2nd InitFromArchive2 raises
        'Unable to open file', R0-verified). Reinit() discards prior RESULTS so the
        next solve is a cold start; the evaluator re-sets ALL inputs each call, so
        every evaluation is independent of the previous one (verified: back-to-
        baseline reproduces exactly, and it recovers cleanly after a broken run)."""
        if not self.mock and self._app is not None:
            try:
                self._app.Reinit()
            except Exception as err:
                diag_log.warning("Aspen Reinit failed: %s", err)

    def resolve_node(self, path):
        """Return the tree node at path, or None (for empirical path verification)."""
        try:
            return self._app.Tree.FindNode(path)
        except Exception as err:
            diag_log.warning("Aspen FindNode failed for %s: %s", path, err)
            return None

    def _node_val(self, path):
        n = self.resolve_node(path)
        if n is None:
            return None
        try:
            return n.Value
        except Exception as err:
            diag_log.warning("Aspen read failed for %s: %s", path, err)
            return None

    def _run_with_watchdog(self):
        """Run the engine on the MAIN thread (correct COM apartment; calling from a
        worker thread raises 'CoInitialize has not been called', R0-verified). A
        daemon TIMER thread kills the engine process on hang and makes NO COM call."""
        import threading
        timeout = self.config["optimizer"]["run_timeout_s"]
        done = threading.Event()
        killed = {"v": False}
        def _watchdog():
            if not done.wait(timeout):
                killed["v"] = True
                diag_log.error("Aspen run exceeded %ss; killing aspenplus.exe", timeout)
                os.system("taskkill /F /IM aspenplus.exe >nul 2>&1")
        wd = threading.Thread(target=_watchdog, daemon=True)
        wd.start()
        try:
            self._app.Engine.Run2()                       # synchronous (blocks)
            ok, err = True, None
        except Exception as e:                            # no bare except
            ok, err = False, f"engine error: {e}"
        finally:
            done.set()
        if killed["v"]:
            self._app = None                              # force re-dispatch+reopen
            self._opened = None
            return False, "timeout"
        return (True, None) if ok else (False, err)

    def _aspen_eval(self, decision, z):
        try:
            self._ensure_open()                           # open baseline once
            self._set_feed_z(z)                           # feed drift + decision vars
            self._set_decision(decision)
            self._apply_tray_efficiency()                 # D3 model-fidelity knob (eta<1)
            self.reset_to_baseline()                      # Reinit cold-start (AFTER set)
            ok, run_err = self._run_with_watchdog()
            if not ok:
                return {"converged": False, "error": run_err,
                        **{k: None for k in OUTPUT_KEYS}}
            if not self._converged():
                return {"converged": False, "error": "Aspen reported non-convergence",
                        **{k: None for k in OUTPUT_KEYS}}
            return self._read_outputs()
        except Exception as err:                          # log, never swallow
            diag_log.error("Aspen evaluation failed (decision=%s z=%.3f): %s",
                           decision, z, err)
            return {"converged": False, "error": f"{type(err).__name__}: {err}",
                    **{k: None for k in OUTPUT_KEYS}}

    def _set_feed_z(self, z):
        # Feed stream "FEED" methanol/water molar flows (z*F, (1-z)*F); total = F.
        for comp, frac in (("METHANOL", z), ("WATER", 1.0 - z)):
            n = self.resolve_node(rf"\Data\Streams\FEED\Input\FLOW\MIXED\{comp}")
            if n is None:
                raise RuntimeError(f"Unresolved Aspen feed node for {comp}")
            n.Value = frac * self.F

    def _set_decision(self, decision):
        for dv in self.config["decision_vars"]:
            if dv["name"] not in decision:
                continue
            val = decision[dv["name"]]
            if dv["name"] == "DtoF":                      # set distillate molar rate
                val = float(val) * self.F
            if dv["type"] == "int":
                val = int(round(val))
            n = self.resolve_node(dv["aspen_path"])
            if n is None:
                raise RuntimeError(f"Unresolved Aspen node for {dv['name']}: "
                                   f"{dv['aspen_path']}")
            n.Value = val

    def _apply_tray_efficiency(self):
        """Honor FCO_TRAY_EFFICIENCY (D3 model-fidelity knob) on the REAL Aspen path.
        eta==1.0 is the equilibrium-stage baseline the RadFrac already runs, so it is a
        NO-OP and every ordinary real run (optimization + verification) is untouched.
        For eta<1 we must set a uniform RadFrac Murphree VAPOR efficiency on the column;
        if no verified efficiency node is configured we RAISE -- we NEVER silently fall
        back to equilibrium, which would make a degraded-fidelity sweep falsely report
        zero model-relative breaches (the exact silent-eta=1 trap the review flagged)."""
        eta = float(os.environ.get("FCO_TRAY_EFFICIENCY", "1.0"))
        if abs(eta - 1.0) < 1e-9:
            return
        path = (self.config["system"].get("fidelity") or {}).get("murphree_vap_eff_path")
        if not path:
            raise RuntimeError(
                "FCO_TRAY_EFFICIENCY=%g requested on the real-Aspen path but "
                "config['system']['fidelity']['murphree_vap_eff_path'] is not set "
                "(pending R0 verification of the RadFrac Murphree vapor-efficiency tree "
                "node). "
                "R0-verify and configure the node before a real-Aspen sweep." % eta)
        n = self.resolve_node(path)
        if n is None:
            raise RuntimeError(
                "Unresolved RadFrac Murphree-efficiency node %s for FCO_TRAY_EFFICIENCY="
                "%g; refusing to run at silent equilibrium fidelity." % (path, eta))
        n.Value = eta

    def _converged(self):
        """Genuine convergence: RadFrac block status BLKSTAT == 0 (R0-verified:
        0 = converged, 1 = non-converged). Anything else => not converged."""
        n = self.resolve_node(self._OUT_PATHS["blkstat"])
        if n is not None and n.Value is not None:
            try:
                return int(n.Value) == 0
            except (TypeError, ValueError):
                return str(n.Value).lower().startswith(("conv", "ok"))
        return False

    def _flooding(self):
        """Fixed-cross-section vapor-throughput flooding proxy from the REAL Aspen
        stage vapor profile: max stage vapor flow / v_flood_ref. Monotonic in
        boilup; NOT rigorous tray-rating %flood (documented in BUILD_NOTES)."""
        vp = self.resolve_node(self._OUT_PATHS["vap_profile"])
        if vp is None:
            return None
        try:
            vals = [vp.Elements.Item(i).Value for i in range(vp.Elements.Count)]
            vmax = max(float(v) for v in vals if v is not None)
        except Exception as err:
            diag_log.warning("VAP_FLOW profile read failed: %s", err)
            return None
        return vmax / self.config["system"]["hydraulics"]["v_flood_ref_kmol_h"]

    def _read_outputs(self):
        hyd    = self.config["system"]["hydraulics"]
        duty_w = self._node_val(self._OUT_PATHS["reboiler_duty"])
        cond_w = self._node_val(self._OUT_PATHS["condenser_duty"])
        xd     = self._node_val(self._OUT_PATHS["dist_methanol_purity"])
        xb     = self._node_val(self._OUT_PATHS["bottoms_methanol"])
        top    = self._node_val(self._OUT_PATHS["top_temp"])
        bot    = self._node_val(self._OUT_PATHS["bottom_temp"])
        # Signed ΔT approaches: the condenser must sit ABOVE the coolant and the
        # reboiler BELOW the steam (a too-hot reboiler -> negative -> infeasible).
        return {
            "converged": True, "error": None,
            "reboiler_duty":        (float(duty_w) / 1000.0) if duty_w is not None else None,
            "condenser_duty":       (float(cond_w) / 1000.0) if cond_w is not None else None,
            "dist_methanol_purity": float(xd) if xd is not None else None,
            "bottoms_methanol":     float(xb) if xb is not None else None,
            "flooding_approach":    self._flooding(),
            "cond_temp_approach":   (float(top) - hyd["cool_util_C"]) if top is not None else None,
            "reb_temp_approach":    (hyd["heat_util_C"] - float(bot)) if bot is not None else None,
        }


# Versioned process-knowledge corpus. Each chunk carries source/version. The
# stale-spec injection (pre-registered): SPEC-DIST-PURITY exists at v1.0 (stale,
# 0.95) AND v2.0 (current, 0.99). A correct system uses v2.0 and/or fails closed
# when a design tuned to v1.0 is verified against the true current constraint.

# ── RAG corpus provenance (author requirement: every chunk traces to a verified
#    real authority) ──────────────────────────────────────────────────────────
# SOURCE_REGISTRY holds the authorities every corpus chunk cites. kind="authority":
# an external, independently-published, web-verified reference (journal paper,
# textbook, handbook, consensus standard, or regulation). kind="constructed": the
# case study's OWN design basis (the stipulated benchmark target VALUES), disclosed
# as illustrative and NEVER the sole source for a chunk. "verified" is the date each
# source's existence/title/edition/DOI/ISBN was web-verified. Per-chunk source keys
# are in _CORPUS_PROVENANCE (after CORPUS) and merged onto each chunk as
# d["provenance"]; the full audit lives in corpus_provenance.json /
# CORPUS_PROVENANCE.md, enforced by tests/test_corpus_provenance_cs2.py.
SOURCE_REGISTRY = {
    # ── separation-design methods: the Fenske-Underwood-Gilliland-Kirkbride (FUGK)
    #    shortcut, the McCabe-Thiele graphical method, O'Connell tray efficiency ──
    "FENSKE": {"kind": "authority", "type": "journal", "verified": "2026-06-03",
        "url": "https://doi.org/10.1021/ie50269a003",
        "citation": "M. R. Fenske, Fractionation of Straight-Run Pennsylvania Gasoline, Ind. Eng. Chem. 24(5), 482-485 (1932) — minimum number of stages at total reflux (Fenske equation)"},
    "UNDERWOOD": {"kind": "authority", "type": "journal", "verified": "2026-06-03",
        "url": "https://scholar.google.com/scholar_lookup?title=Fractional+Distillation+of+Multicomponent+Mixtures&author=A.J.V.+Underwood&publication_year=1948",
        "citation": "A. J. V. Underwood, Fractional Distillation of Multicomponent Mixtures, Chem. Eng. Prog. 44(8), 603-614 (1948) — minimum reflux ratio (Underwood equations); modern treatment in Seader et al. (2011)"},
    "GILLILAND": {"kind": "authority", "type": "journal", "verified": "2026-06-03",
        "url": "https://doi.org/10.1021/ie50369a035",
        "citation": "E. R. Gilliland, Multicomponent Rectification: Estimation of the Number of Theoretical Plates as a Function of the Reflux Ratio, Ind. Eng. Chem. 32(9), 1220-1223 (1940)"},
    "KIRKBRIDE": {"kind": "authority", "type": "journal", "verified": "2026-06-03",
        "url": "https://scholar.google.com/scholar_lookup?title=Process+Design+Procedure+for+Multicomponent+Fractionators&author=C.G.+Kirkbride&publication_year=1944",
        "citation": "C. G. Kirkbride, Process Design Procedure for Multicomponent Fractionators, Petroleum Refiner 23(9), 321-336 (1944) — empirical optimum feed-stage location"},
    "MCCABE_THIELE": {"kind": "authority", "type": "journal", "verified": "2026-06-03",
        "url": "https://doi.org/10.1021/ie50186a023",
        "citation": "W. L. McCabe & E. W. Thiele, Graphical Design of Fractionating Columns, Ind. Eng. Chem. 17(6), 605-611 (1925) — McCabe-Thiele method, operating lines, q-line"},
    "OCONNELL": {"kind": "authority", "type": "journal", "verified": "2026-06-03",
        "url": "https://scholar.google.com/scholar_lookup?title=Plate+Efficiency+of+Fractionating+Columns+and+Absorbers&author=H.E.+O%27Connell&publication_year=1946",
        "citation": "H. E. O'Connell, Plate Efficiency of Fractionating Columns and Absorbers, Trans. AIChE 42, 741-755 (1946) — overall tray-efficiency correlation"},
    "SOUDERS_BROWN": {"kind": "authority", "type": "journal", "verified": "2026-06-03",
        "url": "https://doi.org/10.1021/ie50289a025",
        "citation": "M. Souders & G. G. Brown, Design of Fractionating Columns. I. Entrainment and Capacity, Ind. Eng. Chem. 26(1), 98-103 (1934) — Souders-Brown capacity factor / max vapor velocity"},
    "NRTL": {"kind": "authority", "type": "journal", "verified": "2026-06-03",
        "url": "https://doi.org/10.1002/aic.690140124",
        "citation": "H. Renon & J. M. Prausnitz, Local Compositions in Thermodynamic Excess Functions for Liquid Mixtures, AIChE J. 14(1), 135-144 (1968) — the NRTL activity-coefficient model"},
    "CARLSON": {"kind": "authority", "type": "journal", "verified": "2026-06-03",
        "url": "https://www.semanticscholar.org/paper/Don't-Gamble-With-Physical-Properties-For-Carlson/fe54e7374b25ced213604bae94378a8c355b1a6b",
        "citation": "E. C. Carlson, Don't Gamble With Physical Properties for Simulations, Chem. Eng. Prog. 92(10), 35-46 (1996) — property-method selection for process simulation"},
    # ── reference texts / handbook / standards ──────────────────────────────────
    "SEADER": {"kind": "authority", "type": "textbook", "verified": "2026-06-03",
        "url": "https://www.wiley.com/en-us/Separation+Process+Principles+with+Applications+using+Process+Simulators%2C+3rd+Edition-p-9780470481837",
        "citation": "J. D. Seader, E. J. Henley & D. K. Roper, Separation Process Principles, 3rd ed., Wiley, 2011, ISBN 978-0-470-48183-7"},
    "PERRY": {"kind": "authority", "type": "handbook", "verified": "2026-06-03",
        "url": "https://www.accessengineeringlibrary.com/content/book/9780071834087",
        "citation": "D. W. Green & M. Z. Southard (eds.), Perry's Chemical Engineers' Handbook, 9th ed., McGraw-Hill, 2019, ISBN 978-0-07-183408-7 (Sec. 14 distillation + column internals/hydraulics)"},
    "KISTER": {"kind": "authority", "type": "textbook", "verified": "2026-06-03",
        "url": "https://books.google.com/books/about/Distillation_Design.html?id=0M1TAAAAMAAJ",
        "citation": "H. Z. Kister, Distillation Design, McGraw-Hill, 1992, ISBN 978-0-07-034909-4 (flooding/weeping/downcomer mechanisms, % flood design margin)"},
    "LUYBEN": {"kind": "authority", "type": "textbook", "verified": "2026-06-03",
        "url": "https://onlinelibrary.wiley.com/doi/book/10.1002/9781118510193",
        "citation": "W. L. Luyben, Distillation Design and Control Using Aspen Simulation, 2nd ed., Wiley-AIChE, 2013, ISBN 978-1-118-41143-8 (RadFrac setup/convergence, control structures)"},
    "DOHERTY_MALONE": {"kind": "authority", "type": "textbook", "verified": "2026-06-03",
        "url": "https://search.worldcat.org/title/conceptual-design-of-distillation-systems/oclc/606587023",
        "citation": "M. F. Doherty & M. F. Malone, Conceptual Design of Distillation Systems, McGraw-Hill, 2001, ISBN 978-0-07-118999-6 (azeotropes, distillation boundaries, feasibility)"},
    "ASTM_METHANOL": {"kind": "authority", "type": "consensus-standard", "verified": "2026-06-03",
        "url": "https://store.astm.org/d1152-24.html",
        "citation": "ASTM D1152, Standard Specification for Methanol (Methyl Alcohol); IMPCA Methanol Reference Specifications — Grade AA purity >= 99.85 wt%"},
    "API_521": {"kind": "authority", "type": "consensus-standard", "verified": "2026-06-03",
        "url": "https://www.api.org/products-and-services/standards",
        "citation": "API Standard 521 (ANSI/API Std 521), Pressure-relieving and Depressuring Systems, 7th ed., 2020 — governing relief scenarios (fire, blocked outlet, cooling failure, tube rupture)"},
    "OSHA_PSM": {"kind": "authority", "type": "regulation", "verified": "2026-06-03",
        "url": "https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XVII/part-1910/subpart-H/section-1910.119",
        "citation": "OSHA, Process Safety Management of Highly Hazardous Chemicals, 29 CFR 1910.119(d) — safe upper/lower operating limits"},
    # ── the case study's OWN design basis (disclosed, never a chunk's sole source) ──
    "STUDY_DESIGN": {"kind": "constructed", "type": "constructed-illustrative", "verified": "n/a",
        "url": None,
        "citation": "Case-study design basis — constructed for the methanol/water benchmark column: the stipulated GATED target values (0.99/0.95 distillate, 0.01/0.03 bottoms, 5000 kW reboiler-duty, 5 K approach, 0.80 flood), the decision variables, and the synthetic run logs. Disclosed as illustrative; the validated Aspen model (never a corpus document) enforces every limit; never the sole source for a chunk."},
}

CORPUS = [
    {"id": "SPEC-DIST-PURITY-v2.0", "source_id": "SPEC-DIST-PURITY", "version": "2.0",
     "kind": "constraint", "title": "Distillate methanol purity specification (current)",
     "tags": ["distillate", "purity", "methanol", "spec"],
     "limit": {"variable": "dist_methanol_purity", "relation": ">=", "value": 0.99},
     "text": "CURRENT (v2.0): distillate methanol mole-fraction purity must be at "
             "least 0.99. Supersedes v1.0 (0.95)."},
    {"id": "SPEC-DIST-PURITY-v1.0", "source_id": "SPEC-DIST-PURITY", "version": "1.0",
     "kind": "constraint", "title": "Distillate methanol purity specification (STALE)",
     "tags": ["distillate", "purity", "methanol", "spec", "stale"],
     "limit": {"variable": "dist_methanol_purity", "relation": ">=", "value": 0.95},
     "text": "OUTDATED (v1.0): distillate methanol purity >= 0.95. Retained for "
             "audit; do NOT use — superseded by v2.0 (0.99)."},
    {"id": "SPEC-BOT-PURITY-v2.0", "source_id": "SPEC-BOT-PURITY", "version": "2.0",
     "kind": "constraint", "title": "Bottoms methanol impurity specification (current)",
     "tags": ["bottoms", "impurity", "methanol", "spec"],
     "limit": {"variable": "bottoms_methanol", "relation": "<=", "value": 0.01},
     "text": "CURRENT (v2.0): bottoms methanol mole fraction must be at most 0.01 "
             "(methanol loss). Supersedes v1.0 (0.03)."},
    {"id": "SPEC-BOT-PURITY-v1.0", "source_id": "SPEC-BOT-PURITY", "version": "1.0",
     "kind": "constraint", "title": "Bottoms methanol impurity specification (STALE)",
     "tags": ["bottoms", "impurity", "methanol", "spec", "stale"],
     "limit": {"variable": "bottoms_methanol", "relation": "<=", "value": 0.03},
     "text": "OUTDATED (v1.0): bottoms methanol <= 0.03. Retained for audit; do NOT "
             "use — superseded by v2.0 (0.01). The v1.0 pair (0.95 / 0.03) admits a "
             "lower-reflux, lower-duty design that VIOLATES the current 0.99 / 0.01."},
    {"id": "LIM-FLOODING-v1.0", "source_id": "LIM-FLOODING", "version": "1.0",
     "kind": "constraint", "title": "Column flooding approach limit",
     "tags": ["flooding", "hydraulics", "limit"],
     "limit": {"variable": "flooding_approach", "relation": "<=", "value": 0.80},
     "text": "Maximum fractional approach to flooding is 0.80 (80%); above this the "
             "column hydraulics are unreliable and convergence degrades."},
    {"id": "LIM-FLOODING-v0.9", "source_id": "LIM-FLOODING", "version": "0.9",
     "kind": "constraint", "title": "Column flooding approach limit (STALE)",
     "tags": ["flooding", "hydraulics", "limit", "stale"],
     "limit": {"variable": "flooding_approach", "relation": "<=", "value": 0.90},
     "text": "OUTDATED (v0.9): an older standard allowed up to 0.90 approach to flooding. "
             "Retained for audit; do NOT use - superseded by v1.0 (0.80)."},
    {"id": "LIM-REB-DUTY-v1.0", "source_id": "LIM-REB-DUTY", "version": "1.0",
     "kind": "constraint", "title": "Reboiler duty utility limit",
     "tags": ["reboiler", "duty", "steam", "utility", "limit"],
     "limit": {"variable": "reboiler_duty", "relation": "<=", "value": 5000.0},
     "text": "Reboiler duty must not exceed the 5000 kW LP-steam utility limit."},
    {"id": "LIM-REB-DUTY-v0.9", "source_id": "LIM-REB-DUTY", "version": "0.9",
     "kind": "constraint", "title": "Reboiler duty utility limit (STALE)",
     "tags": ["reboiler", "duty", "steam", "utility", "limit", "stale"],
     "limit": {"variable": "reboiler_duty", "relation": "<=", "value": 5500.0},
     "text": "OUTDATED (v0.9): an older utility contract allowed up to 5500 kW reboiler "
             "duty. Retained for audit; do NOT use - superseded by v1.0 (5000 kW)."},
    {"id": "LIM-DT-MIN-v1.0", "source_id": "LIM-DT-MIN", "version": "1.0",
     "kind": "constraint", "title": "Minimum heat-exchanger temperature approach",
     "tags": ["temperature", "approach", "condenser", "reboiler", "limit"],
     "limit": {"variable": "temp_approach", "relation": ">=", "value": 5.0},
     "text": "Condenser and reboiler temperature approaches must be at least 5 K "
             "for a feasible, controllable exchanger design."},
    {"id": "HEUR-REFLUX-v1.0", "source_id": "HEUR-REFLUX", "version": "1.0",
     "kind": "heuristic", "title": "Reflux ratio vs reboiler duty heuristic",
     "tags": ["reflux", "minimum reflux", "energy", "rationale"],
     "text": "Reboiler duty rises roughly linearly with boilup, hence with reflux. "
             "To minimize energy, operate near the MINIMUM reflux that still meets "
             "both product purity specs (typically 1.05-1.3x minimum reflux); "
             "higher reflux only wastes steam and raises flooding."},
    {"id": "HEUR-FEEDSTAGE-v1.0", "source_id": "HEUR-FEEDSTAGE", "version": "1.0",
     "kind": "heuristic", "title": "Feed-stage placement heuristic",
     "tags": ["feed stage", "Kirkbride", "rationale"],
     "text": "Place the feed near the stage whose composition matches the feed; a "
             "poorly placed feed forces extra reflux/boilup for the same split. "
             "For a ~50/50 feed in a 25-stage column, mid-column (~stage 13) is a "
             "good starting feed location (Kirkbride for refinement)."},
    {"id": "PLAYBOOK-CONV-v1.0", "source_id": "PLAYBOOK-CONV", "version": "1.0",
     "kind": "playbook", "title": "RadFrac convergence playbook",
     "tags": ["convergence", "modeling", "playbook"],
     "text": "If RadFrac fails to converge: verify the distillate-to-feed ratio is "
             "physically feasible for the feed composition, keep reflux above "
             "minimum, avoid >95% flooding, and treat a non-converged run as "
             "INFEASIBLE (never a large objective value)."},
    {"id": "RUNLOG-001-v1.0", "source_id": "RUNLOG-001", "version": "1.0",
     "kind": "run_log", "title": "Prior converged run log",
     "tags": ["run log", "history"],
     "text": "Prior run: reflux 2.5, D:F 0.50, feed stage 13 -> reboiler duty "
             "~1.8 MW, converged, distillate methanol 0.994, bottoms 0.006 (on spec)."},

    # ══ Enrichment: a realistic distillation SOP / spec library ═══════════════════
    # New source_ids ONLY — the 6 GATED specs above (SPEC-DIST-PURITY, SPEC-BOT-PURITY,
    # LIM-FLOODING, LIM-REB-DUTY, LIM-DT-MIN) and their stale versions are UNCHANGED, so
    # the gate's inputs (current_spec of the gated ids) and the mock decision are
    # unchanged -> the study results are unaffected (verified by a full-mock metrics
    # diff). These chunks exercise both retrieval channels with realistic distractors.

    # ── shortcut / separation-design heuristics ──────────────────────────────────
    {"id": "HEUR-FENSKE-v1.0", "source_id": "HEUR-FENSKE", "version": "1.0",
     "kind": "heuristic", "title": "Fenske minimum number of stages",
     "tags": ["minimum stages", "fenske", "total reflux", "shortcut", "relative volatility"],
     "text": "Fenske gives the minimum theoretical stages at total reflux: "
             "Nmin = ln[(xD/(1-xD))((1-xB)/xB)]/ln(alpha). For methanol/water (alpha ~ "
             "3-4) a 0.99/0.01 split needs only a few stages at total reflux; real "
             "columns use 1.5-3x Nmin."},
    {"id": "HEUR-UNDERWOOD-v1.0", "source_id": "HEUR-UNDERWOOD", "version": "1.0",
     "kind": "heuristic", "title": "Underwood minimum reflux ratio",
     "tags": ["minimum reflux", "underwood", "rmin", "shortcut"],
     "text": "Underwood's equations estimate the minimum reflux ratio Rmin from the feed "
             "condition and relative volatility. Operating reflux is chosen as a multiple "
             "of Rmin; below Rmin the separation is infeasible at any number of stages."},
    {"id": "HEUR-GILLILAND-v1.0", "source_id": "HEUR-GILLILAND", "version": "1.0",
     "kind": "heuristic", "title": "Gilliland stages-vs-reflux correlation",
     "tags": ["gilliland", "stages", "reflux", "shortcut", "fug"],
     "text": "The Gilliland correlation links actual reflux/stages to Nmin and Rmin (the "
             "Fenske-Underwood-Gilliland shortcut) for a first estimate before a rigorous "
             "RadFrac run; it is not a substitute for the rigorous solve."},
    {"id": "HEUR-KIRKBRIDE-v1.0", "source_id": "HEUR-KIRKBRIDE", "version": "1.0",
     "kind": "heuristic", "title": "Kirkbride feed-stage location",
     "tags": ["feed stage", "kirkbride", "feed location", "rectifying", "stripping"],
     "text": "Kirkbride estimates the optimal feed-stage ratio (stages above vs below the "
             "feed) from the feed composition and product purities. For a ~50/50 "
             "methanol/water feed in a 25-stage column the optimum sits near mid-column; "
             "a poorly placed feed wastes reflux and boilup."},
    {"id": "HEUR-RELVOL-v1.0", "source_id": "HEUR-RELVOL", "version": "1.0",
     "kind": "heuristic", "title": "Relative volatility and separation difficulty",
     "tags": ["relative volatility", "alpha", "difficulty", "easy separation"],
     "text": "Separation difficulty scales inversely with relative volatility alpha: "
             "alpha>>1 is easy (few stages, low reflux), alpha->1 is hard. Methanol/water "
             "is a moderately easy, non-azeotropic separation."},
    {"id": "HEUR-RR-OPTIMUM-v1.0", "source_id": "HEUR-RR-OPTIMUM", "version": "1.0",
     "kind": "heuristic", "title": "Economic optimum reflux ratio",
     "tags": ["reflux", "optimum", "economic", "energy", "rmin"],
     "text": "The economic optimum reflux is typically 1.05-1.3x minimum reflux: higher "
             "reflux raises reboiler duty (operating cost) for diminishing stage savings "
             "(capital). To minimise energy, operate near the minimum reflux that still "
             "meets both product specs."},
    {"id": "HEUR-MCCABE-THIELE-v1.0", "source_id": "HEUR-MCCABE-THIELE", "version": "1.0",
     "kind": "heuristic", "title": "McCabe-Thiele graphical analysis",
     "tags": ["mccabe-thiele", "graphical", "operating line", "q-line", "binary"],
     "text": "For a binary system the McCabe-Thiele diagram shows the stage requirement "
             "from the equilibrium curve, the rectifying/stripping operating lines and "
             "the q-line; a pinch near the feed indicates near-minimum reflux."},
    {"id": "HEUR-TRAY-EFF-v1.0", "source_id": "HEUR-TRAY-EFF", "version": "1.0",
     "kind": "heuristic", "title": "Tray efficiency (O'Connell)",
     "tags": ["tray efficiency", "oconnell", "actual trays", "murphree"],
     "text": "Overall column efficiency (O'Connell) relates theoretical to actual trays "
             "via the feed viscosity-volatility product; methanol/water typically gives "
             "60-80% overall efficiency, so actual trays exceed theoretical stages."},

    # ── hydraulics / sizing (supporting + distractor limits; NON-gated) ──────────
    {"id": "HEUR-PRESSURE-SELECT-v1.0", "source_id": "HEUR-PRESSURE-SELECT", "version": "1.0",
     "kind": "heuristic", "title": "Column operating-pressure selection",
     "tags": ["pressure", "condenser", "coolant", "selection"],
     "text": "Choose the lowest pressure that still lets cooling water (or air) condense "
             "the overhead: this improves relative volatility and lowers reboiler "
             "temperature. Methanol/water near 1 atm condenses with cooling water; "
             "vacuum is unnecessary and pressure raises the bottoms temperature."},
    {"id": "LIM-COLUMN-DP-v1.0", "source_id": "LIM-COLUMN-DP", "version": "1.0",
     "kind": "constraint", "title": "Column pressure-drop limit (STALE)",
     "tags": ["pressure drop", "hydraulics", "dp", "limit", "stale"],
     "limit": {"variable": "column_dp_bar", "relation": "<=", "value": 0.20},
     "text": "OUTDATED (v1.0): total column pressure drop <= 0.20 bar. Superseded by "
             "v2.0 (0.15). A NON-gated distractor for retrieval + version-handling tests."},
    {"id": "LIM-COLUMN-DP-v2.0", "source_id": "LIM-COLUMN-DP", "version": "2.0",
     "kind": "constraint", "title": "Column pressure-drop limit (current)",
     "tags": ["pressure drop", "hydraulics", "dp", "limit"],
     "limit": {"variable": "column_dp_bar", "relation": "<=", "value": 0.15},
     "text": "CURRENT (v2.0): total column pressure drop <= 0.15 bar for bottoms "
             "temperature and tray stability. Not a gated spec in this study."},
    {"id": "LIM-COND-DUTY-v1.0", "source_id": "LIM-COND-DUTY", "version": "1.0",
     "kind": "constraint", "title": "Condenser cooling-duty limit",
     "tags": ["condenser", "cooling", "duty", "utility", "limit"],
     "limit": {"variable": "condenser_duty_kW", "relation": "<=", "value": 5500.0},
     "text": "Condenser cooling duty must not exceed the cooling-water exchanger limit; "
             "it tracks the reboiler duty closely at total condensation. Not gated here."},
    {"id": "LIM-WEEPING-v1.0", "source_id": "LIM-WEEPING", "version": "1.0",
     "kind": "constraint", "title": "Minimum vapor load (weeping) limit",
     "tags": ["weeping", "minimum vapor", "turndown", "hydraulics", "limit"],
     "limit": {"variable": "flooding_approach", "relation": ">=", "value": 0.20},
     "text": "Below ~20% of flood, trays weep and efficiency collapses - the lower "
             "hydraulic bound complementing the 0.80 flooding ceiling. Sets the turndown "
             "window; not separately gated in this MVP."},
    {"id": "LIM-DOWNCOMER-v1.0", "source_id": "LIM-DOWNCOMER", "version": "1.0",
     "kind": "constraint", "title": "Downcomer flooding limit",
     "tags": ["downcomer", "flooding", "liquid load", "hydraulics", "limit"],
     "limit": {"variable": "downcomer_flood", "relation": "<=", "value": 0.80},
     "text": "Downcomer (liquid) flooding is a separate mechanism from jet (vapor) "
             "flooding; both stay below ~80%. The study uses a vapor-throughput flooding "
             "proxy; rigorous tray rating would add this term."},
    {"id": "LIM-COLUMN-DIAMETER-v1.0", "source_id": "LIM-COLUMN-DIAMETER", "version": "1.0",
     "kind": "constraint", "title": "Column diameter / capacity (Souders-Brown)",
     "tags": ["diameter", "sizing", "souders-brown", "capacity"],
     "limit": {"variable": "diameter_m", "relation": "<=", "value": 2.0},
     "text": "Column diameter from the Souders-Brown capacity factor at the highest-load "
             "stage; larger diameter handles more vapor before flooding (a capital "
             "trade-off). Sizing context; not gated."},

    # ── cross-system distractor specs (force component-level disambiguation) ──────
    {"id": "SPEC-ETHANOL-WATER-v1.0", "source_id": "SPEC-ETHANOL-WATER", "version": "1.0",
     "kind": "constraint", "title": "Ethanol/water distillate purity (DIFFERENT system)",
     "tags": ["ethanol", "purity", "distillate", "azeotrope", "spec"],
     "limit": {"variable": "dist_ethanol_purity", "relation": ">=", "value": 0.95},
     "text": "Ethanol/water distillate >= 0.95 (95 wt%); the ethanol-water azeotrope "
             "(~95.6 wt%, ~0.894 mole fraction at 1 atm) caps ordinary distillation, so "
             "higher purity needs an entrainer or pressure-swing scheme. A DIFFERENT "
             "system from this study's methanol/water, so retrieval must disambiguate by "
             "component, not just on the word 'purity'."},
    {"id": "SPEC-BENZENE-TOLUENE-v1.0", "source_id": "SPEC-BENZENE-TOLUENE", "version": "1.0",
     "kind": "constraint", "title": "Benzene/toluene split (DIFFERENT system)",
     "tags": ["benzene", "toluene", "purity", "peng-robinson", "spec"],
     "limit": {"variable": "dist_benzene_purity", "relation": ">=", "value": 0.99},
     "text": "Benzene/toluene is a near-ideal split (Peng-Robinson), unlike the non-ideal "
             "methanol/water of this study. Distractor for retrieval."},

    # ── property-method / modeling guidance ──────────────────────────────────────
    {"id": "HEUR-PROPERTY-NRTL-v1.0", "source_id": "HEUR-PROPERTY-NRTL", "version": "1.0",
     "kind": "heuristic", "title": "NRTL for non-ideal liquids",
     "tags": ["nrtl", "property method", "activity coefficient", "non-ideal", "methanol"],
     "text": "Methanol/water is a polar, non-ideal mixture, so an activity-coefficient "
             "model (NRTL or UNIQUAC) with fitted binary parameters is appropriate; a "
             "cubic EOS (Peng-Robinson) suits near-ideal hydrocarbons. The wrong method "
             "mis-predicts the VLE and the duty."},
    {"id": "HEUR-PROPERTY-SELECT-v1.0", "source_id": "HEUR-PROPERTY-SELECT", "version": "1.0",
     "kind": "heuristic", "title": "Property-method selection guide",
     "tags": ["property method", "selection", "eos", "activity coefficient", "polar"],
     "text": "Selection: polar/non-ideal -> NRTL/UNIQUAC/Wilson; electrolytes -> "
             "ELECNRTL; non-polar hydrocarbons -> PENG-ROB/SRK. Confirm against binary "
             "VLE data; a wrong method is a common source of silently wrong duties."},
    {"id": "PLAYBOOK-INIT-v1.0", "source_id": "PLAYBOOK-INIT", "version": "1.0",
     "kind": "playbook", "title": "RadFrac initialization strategy",
     "tags": ["initialization", "convergence", "radfrac", "playbook"],
     "text": "If RadFrac struggles: start from a converged nearby case, give reasonable "
             "reflux/boilup estimates, tighten specs gradually, and Reinit between "
             "independent runs to avoid carry-over."},
    {"id": "PLAYBOOK-AZEOTROPE-v1.0", "source_id": "PLAYBOOK-AZEOTROPE", "version": "1.0",
     "kind": "playbook", "title": "Azeotrope feasibility check",
     "tags": ["azeotrope", "distillation boundary", "feasibility", "playbook"],
     "text": "Before targeting a purity, check for an azeotrope that caps ordinary "
             "distillation. Methanol/water has NO azeotrope, so 0.99+ is reachable; "
             "ethanol/water DOES (~95.6 wt%, ~0.894 mole fraction at 1 atm), needing an "
             "entrainer or pressure-swing scheme."},

    # ── safety / operability ─────────────────────────────────────────────────────
    {"id": "SAFETY-RELIEF-v1.0", "source_id": "SAFETY-RELIEF", "version": "1.0",
     "kind": "playbook", "title": "Relief and overpressure protection",
     "tags": ["relief", "safety", "overpressure", "psv", "operability"],
     "text": "Columns need overpressure protection sized for the governing scenario "
             "(blocked outlet, cooling failure, fire). A steady-state optimisation fixes "
             "pressure; a real design must size relief separately. Out of scope here but "
             "required for a real unit."},
    {"id": "OPER-CONTROL-v1.0", "source_id": "OPER-CONTROL", "version": "1.0",
     "kind": "playbook", "title": "Column control structure",
     "tags": ["control", "operability", "lv", "db", "dual composition"],
     "text": "Common pairings: reflux/boilup (LV) or distillate/boilup (DB). Single-end "
             "composition control with the other end on a fixed reflux ratio is typical; "
             "dual-composition control saves energy but is harder to tune."},
    {"id": "OPER-TURNDOWN-v1.0", "source_id": "OPER-TURNDOWN", "version": "1.0",
     "kind": "playbook", "title": "Turndown and the operability window",
     "tags": ["turndown", "operability", "weeping", "flooding", "window"],
     "text": "The stable operating window runs from weeping (low vapor) to flooding "
             "(high vapor). A robust design keeps the nominal point comfortably inside so "
             "feed-rate turndown does not cross either bound."},
    {"id": "HEUR-FEED-PREHEAT-v1.0", "source_id": "HEUR-FEED-PREHEAT", "version": "1.0",
     "kind": "heuristic", "title": "Feed thermal condition (q)",
     "tags": ["feed", "preheat", "q-line", "thermal condition", "vapor fraction"],
     "text": "The feed thermal condition q shifts duty between reboiler and condenser and "
             "moves the q-line. A saturated-liquid feed (q=1, this study) is a common "
             "baseline; preheating to partial vapor lowers reboiler duty at the cost of "
             "feed-exchanger duty."},

    # ── additional run logs (history) ────────────────────────────────────────────
    {"id": "RUNLOG-002-v1.0", "source_id": "RUNLOG-002", "version": "1.0",
     "kind": "run_log", "title": "Prior run log - near energy minimum",
     "tags": ["run log", "history", "low reflux"],
     "text": "Prior run: reflux 0.8, D:F 0.50, feed stage 13 -> reboiler duty ~0.89 MW, "
             "converged, distillate methanol 0.998, bottoms 0.002 (on the current "
             "0.99/0.01 spec); near the energy-minimum boundary."},
    {"id": "RUNLOG-003-v1.0", "source_id": "RUNLOG-003", "version": "1.0",
     "kind": "run_log", "title": "Prior run log - feed drift recovery",
     "tags": ["run log", "history", "feed drift"],
     "text": "Prior run: feed methanol drifted to 0.45; the fixed 0.50 design lost the "
             "bottoms spec until D:F was re-optimised to ~0.45 -> converged on spec. "
             "Illustrates the mass-balance pinning of the cut to the feed composition."},
]


# ── per-chunk provenance: which verified authorities each chunk traces to, how it
#    was derived, and a cleanliness grade. A: cleanly sourced textbook/standard
#    method (text essentially unchanged). B: real principle/method with
#    study-illustrative numbers (the dominant, honest class). C: constructed/
#    synthetic study artifact, disclosed as illustrative. Every chunk lists >=1
#    kind="authority" source; STUDY_DESIGN is never a chunk's sole source. This is
#    metadata only — merged onto each chunk as d["provenance"] below, NOT indexed
#    (the vectoriser reads title+text+tags) and NOT returned by retrieve() (see
#    _passage), so the locked study is behaviour-identical. ──────────────────────
_CORPUS_PROVENANCE = {
    # gated specs/limits — real spec/limit practice, study-stipulated VALUES
    "SPEC-DIST-PURITY-v2.0": {"sources": ["ASTM_METHANOL", "STUDY_DESIGN"], "derivation": "constructed-illustrative", "cleanliness": "B"},
    "SPEC-DIST-PURITY-v1.0": {"sources": ["ASTM_METHANOL", "STUDY_DESIGN"], "derivation": "constructed-illustrative", "cleanliness": "B"},
    "SPEC-BOT-PURITY-v2.0":  {"sources": ["SEADER", "STUDY_DESIGN"], "derivation": "constructed-illustrative", "cleanliness": "B"},
    "SPEC-BOT-PURITY-v1.0":  {"sources": ["SEADER", "STUDY_DESIGN"], "derivation": "constructed-illustrative", "cleanliness": "B"},
    "LIM-FLOODING-v1.0":     {"sources": ["KISTER", "PERRY", "STUDY_DESIGN"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "LIM-FLOODING-v0.9":     {"sources": ["KISTER", "PERRY", "STUDY_DESIGN"], "derivation": "constructed-illustrative", "cleanliness": "B"},
    "LIM-REB-DUTY-v1.0":     {"sources": ["OSHA_PSM", "STUDY_DESIGN"], "derivation": "constructed-illustrative", "cleanliness": "B"},
    "LIM-REB-DUTY-v0.9":     {"sources": ["OSHA_PSM", "STUDY_DESIGN"], "derivation": "constructed-illustrative", "cleanliness": "B"},
    "LIM-DT-MIN-v1.0":       {"sources": ["PERRY", "STUDY_DESIGN"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    # original heuristics / playbook / run log
    "HEUR-REFLUX-v1.0":      {"sources": ["SEADER", "UNDERWOOD"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "HEUR-FEEDSTAGE-v1.0":   {"sources": ["KIRKBRIDE", "SEADER"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "PLAYBOOK-CONV-v1.0":    {"sources": ["LUYBEN", "SEADER"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "RUNLOG-001-v1.0":       {"sources": ["STUDY_DESIGN", "SEADER"], "derivation": "constructed-illustrative", "cleanliness": "C"},
    # shortcut / separation-design methods (clean textbook methods)
    "HEUR-FENSKE-v1.0":      {"sources": ["FENSKE", "SEADER"], "derivation": "paraphrased", "cleanliness": "A"},
    "HEUR-UNDERWOOD-v1.0":   {"sources": ["UNDERWOOD", "SEADER"], "derivation": "paraphrased", "cleanliness": "A"},
    "HEUR-GILLILAND-v1.0":   {"sources": ["GILLILAND", "SEADER"], "derivation": "paraphrased", "cleanliness": "A"},
    "HEUR-KIRKBRIDE-v1.0":   {"sources": ["KIRKBRIDE", "SEADER"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "HEUR-RELVOL-v1.0":      {"sources": ["SEADER"], "derivation": "paraphrased", "cleanliness": "A"},
    "HEUR-RR-OPTIMUM-v1.0":  {"sources": ["SEADER"], "derivation": "paraphrased", "cleanliness": "A"},
    "HEUR-MCCABE-THIELE-v1.0": {"sources": ["MCCABE_THIELE", "SEADER"], "derivation": "paraphrased", "cleanliness": "A"},
    "HEUR-TRAY-EFF-v1.0":    {"sources": ["OCONNELL", "SEADER"], "derivation": "paraphrased", "cleanliness": "A"},
    # hydraulics / sizing
    "HEUR-PRESSURE-SELECT-v1.0": {"sources": ["SEADER", "KISTER"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "LIM-COLUMN-DP-v1.0":    {"sources": ["KISTER", "STUDY_DESIGN"], "derivation": "constructed-illustrative", "cleanliness": "B"},
    "LIM-COLUMN-DP-v2.0":    {"sources": ["KISTER", "STUDY_DESIGN"], "derivation": "constructed-illustrative", "cleanliness": "B"},
    "LIM-COND-DUTY-v1.0":    {"sources": ["PERRY", "STUDY_DESIGN"], "derivation": "constructed-illustrative", "cleanliness": "B"},
    "LIM-WEEPING-v1.0":      {"sources": ["KISTER", "PERRY"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "LIM-DOWNCOMER-v1.0":    {"sources": ["KISTER", "PERRY"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "LIM-COLUMN-DIAMETER-v1.0": {"sources": ["SOUDERS_BROWN", "KISTER"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    # cross-system distractors (real facts about OTHER systems)
    "SPEC-ETHANOL-WATER-v1.0": {"sources": ["DOHERTY_MALONE", "PERRY"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "SPEC-BENZENE-TOLUENE-v1.0": {"sources": ["SEADER", "PERRY"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    # property-method / modeling guidance
    "HEUR-PROPERTY-NRTL-v1.0": {"sources": ["NRTL", "CARLSON"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "HEUR-PROPERTY-SELECT-v1.0": {"sources": ["CARLSON", "SEADER"], "derivation": "paraphrased", "cleanliness": "A"},
    "PLAYBOOK-INIT-v1.0":    {"sources": ["LUYBEN", "SEADER"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "PLAYBOOK-AZEOTROPE-v1.0": {"sources": ["DOHERTY_MALONE", "SEADER"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    # safety / operability
    "SAFETY-RELIEF-v1.0":    {"sources": ["API_521", "OSHA_PSM"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "OPER-CONTROL-v1.0":     {"sources": ["LUYBEN"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "OPER-TURNDOWN-v1.0":    {"sources": ["KISTER", "PERRY"], "derivation": "synthesized-from-source", "cleanliness": "B"},
    "HEUR-FEED-PREHEAT-v1.0": {"sources": ["SEADER", "MCCABE_THIELE"], "derivation": "paraphrased", "cleanliness": "B"},
    # additional run logs (synthetic study records)
    "RUNLOG-002-v1.0":       {"sources": ["STUDY_DESIGN", "UNDERWOOD"], "derivation": "constructed-illustrative", "cleanliness": "C"},
    "RUNLOG-003-v1.0":       {"sources": ["STUDY_DESIGN", "SEADER"], "derivation": "constructed-illustrative", "cleanliness": "C"},
}

# ── flag-gated retrieval DISTRACTORS (FCO_DISTRACTORS=1; OFF by default, so the base
# 40-chunk corpus and the locked study are byte-for-byte unchanged). Confusable-but-
# IRRELEVANT admin/maintenance chunks (they mention column/reflux/duty/tray/purity, so a
# keyword retriever is tempted), but their unique source_ids are NEVER in any rag_eval
# relevant set, so a better (dense/hybrid) retriever ranks them below the real specs and
# heuristics -> lets dense-vs-lexical separate. Symmetric with case1_reactor CS1_DISTRACTORS;
# provenance-tracked like every chunk.
CS2_DISTRACTORS = [
    {"id": "DIST-C2-001", "source_id": "DIST-C2-001", "version": "1.0", "kind": "distractor",
     "title": "Column insulation inspection schedule (routine)",
     "tags": ["column", "reboiler", "condenser", "insulation", "maintenance"],
     "text": "Routine mechanical-integrity inspection cadence for the column, reboiler and "
             "condenser insulation/lagging; record damage and schedule recladding. Maintenance "
             "scheduling only -- NOT a design spec, operability limit, or operating action."},
    {"id": "DIST-C2-002", "source_id": "DIST-C2-002", "version": "1.0", "kind": "distractor",
     "title": "Reflux pump seal lubrication and PM",
     "tags": ["reflux", "pump", "seal", "lubrication", "maintenance"],
     "text": "Preventive-maintenance lubrication intervals for the reflux and bottoms pump "
             "bearings and mechanical seals; grease grade and work-order cadence. Routine "
             "maintenance only -- unrelated to the reflux RATIO or any separation specification."},
    {"id": "DIST-C2-003", "source_id": "DIST-C2-003", "version": "1.0", "kind": "distractor",
     "title": "Operator training and competency records (distillation unit)",
     "tags": ["operator", "training", "competency", "reflux", "purity"],
     "text": "Operator training matrix and re-certification cadence for the distillation unit, "
             "including refresher modules on reflux-ratio control and product-purity sampling. "
             "Administrative training records only -- NOT an operating procedure or design limit."},
    {"id": "DIST-C2-004", "source_id": "DIST-C2-004", "version": "1.0", "kind": "distractor",
     "title": "Management-of-Change paperwork routing (column setpoints)",
     "tags": ["MOC", "management of change", "setpoint", "approval", "column"],
     "text": "Administrative routing and approval signatures for a management-of-change request "
             "that adjusts column reflux or duty setpoints; form numbers and review queue. "
             "Document-control workflow only -- NOT a specification, limit, or operating action."},
    {"id": "DIST-C2-005", "source_id": "DIST-C2-005", "version": "1.0", "kind": "distractor",
     "title": "Personal protective equipment inventory (column area)",
     "tags": ["PPE", "inventory", "column", "safety", "procurement"],
     "text": "PPE stock levels and reorder points for the distillation area: gloves, face shields, "
             "and chemical aprons rated for methanol service. Procurement/inventory administration "
             "only -- unrelated to any purity spec, flooding limit, or duty constraint."},
    {"id": "DIST-C2-006", "source_id": "DIST-C2-006", "version": "1.0", "kind": "distractor",
     "title": "Tray and gasket spare-parts procurement specification",
     "tags": ["tray", "gasket", "spares", "procurement", "valve tray"],
     "text": "Approved-supplier list, materials, and minimum on-site inventory for column tray "
             "valves and flange gaskets. Stores/procurement administration only -- NOT a tray "
             "hydraulic limit (flooding/weeping) or a separation design heuristic."},
    {"id": "DIST-C2-007", "source_id": "DIST-C2-007", "version": "1.0", "kind": "distractor",
     "title": "Laboratory sample-logging procedure (product streams)",
     "tags": ["lab", "sample", "logging", "distillate", "bottoms", "purity"],
     "text": "Chain-of-custody and logbook procedure for grab samples of the distillate and "
             "bottoms streams sent to the lab. Records-management administration only -- it states "
             "NO purity threshold and is NOT a product specification."},
    {"id": "DIST-C2-008", "source_id": "DIST-C2-008", "version": "1.0", "kind": "distractor",
     "title": "Pressure-gauge calibration schedule (column instruments)",
     "tags": ["pressure", "gauge", "calibration", "instrument", "schedule"],
     "text": "Calibration interval and drift-recording schedule for the column pressure gauges and "
             "transmitters per the instrument-management program. Routine instrument maintenance "
             "only -- NOT a column pressure-drop limit or an operating setpoint."},
]
_CORPUS_PROVENANCE.update({
    "DIST-C2-001": {"sources": ["KISTER", "OSHA_PSM"], "derivation": "constructed-illustrative", "cleanliness": "C"},
    "DIST-C2-002": {"sources": ["KISTER", "PERRY"], "derivation": "constructed-illustrative", "cleanliness": "C"},
    "DIST-C2-003": {"sources": ["OSHA_PSM", "LUYBEN"], "derivation": "constructed-illustrative", "cleanliness": "C"},
    "DIST-C2-004": {"sources": ["OSHA_PSM"], "derivation": "constructed-illustrative", "cleanliness": "C"},
    "DIST-C2-005": {"sources": ["OSHA_PSM"], "derivation": "constructed-illustrative", "cleanliness": "C"},
    "DIST-C2-006": {"sources": ["KISTER", "PERRY"], "derivation": "constructed-illustrative", "cleanliness": "C"},
    "DIST-C2-007": {"sources": ["OSHA_PSM", "PERRY"], "derivation": "constructed-illustrative", "cleanliness": "C"},
    "DIST-C2-008": {"sources": ["PERRY", "OSHA_PSM"], "derivation": "constructed-illustrative", "cleanliness": "C"},
})
if os.environ.get("FCO_DISTRACTORS"):
    CORPUS.extend(CS2_DISTRACTORS)        # re-run only; default OFF keeps the locked 40-chunk corpus


# Attach provenance to each chunk (metadata only). A chunk with no record gets a
# loud MISSING marker so tests/test_corpus_provenance_cs2.py fails closed rather than
# silently shipping an unsourced chunk.
for _chunk in CORPUS:
    _chunk["provenance"] = _CORPUS_PROVENANCE.get(
        _chunk["id"], {"sources": [], "derivation": "MISSING", "cleanliness": "?"})


# ══ lexical retrieval helpers (offline; no embedding backend needed) ═══════════
_RAG_STOP = None   # lazy sklearn English stopword set


def _rag_tokens(text):
    """Lowercase alphanumeric tokenization with English stopword removal (no
    aggressive stemming — it hurts short domain terms like 'gas'/'tray'). Feeds BM25."""
    global _RAG_STOP
    if _RAG_STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _RAG_STOP = ENGLISH_STOP_WORDS
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(t) >= 2 and t not in _RAG_STOP]


# Small, conservative domain synonym map for QUERY expansion only (a recall booster;
# expanding the query, never the documents, keeps it low-risk).
_RAG_SYNONYMS = {
    "rr": ["reflux"], "reflux": ["rr"],
    "duty": ["energy", "steam", "reboiler"], "energy": ["duty"],
    "minimize": ["minimum"], "minimise": ["minimum"], "minimum": ["min"],
    "flood": ["flooding"], "flooding": ["flood"],
    "tray": ["stage", "plate"], "stage": ["tray"], "stages": ["tray"],
    "purity": ["composition", "molefrac"], "spec": ["specification", "limit"],
    "converge": ["convergence"], "convergence": ["converge"],
    "azeotrope": ["azeotropic"], "property": ["method", "thermodynamic"],
}


class _BM25:
    """Minimal Okapi BM25 over pre-tokenized documents (pure Python; no new deps)."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b, self.docs = k1, b, docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = (sum(len(d) for d in docs_tokens) / self.N) if self.N else 0.0
        df, self.tf = defaultdict(int), []
        for d in docs_tokens:
            counts = defaultdict(int)
            for t in d:
                counts[t] += 1
            self.tf.append(counts)
            for t in counts:
                df[t] += 1
        self.idf = {t: math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
                    for t, n in df.items()}

    def scores(self, q_tokens):
        out = []
        for i, counts in enumerate(self.tf):
            dl, s = len(self.docs[i]), 0.0
            for t in q_tokens:
                f = counts.get(t, 0)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0)))
            out.append(s)
        return out


class RAGEngine:
    """
    Dual-channel retriever (Phase 3):
      - SPARSE constraint channel: exact/keyword match over 'constraint' chunks
        and their numerical limits, returning the CURRENT version for a source_id
        and flagging version conflicts via metadata (stale-spec mechanism).
      - DENSE rationale channel: TF-IDF + BM25 fused with a real embedding model
        (fastembed, enabled by RAG_DENSE=1; see core/embeddings.py) via reciprocal
        rank fusion. Without RAG_DENSE it is the lexical bm25+tfidf hybrid. The active
        backend is reported (self.dense_backend) and recorded in the manifest.
    Every returned passage carries source/version.
    """
    def __init__(self, docs=None):
        self.docs = docs if docs is not None else CORPUS
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        self._cos = cosine_similarity
        corpus_txt = [f"{d['title']} {d['text']} {' '.join(d.get('tags', []))*3}"
                      for d in self.docs]
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=1.0)
        self._mat = self._vec.fit_transform(corpus_txt)
        # BM25 lexical channel over the same (tag-weighted) text; fused with TF-IDF in
        # retrieve() via reciprocal rank fusion -> stronger than raw TF-IDF cosine.
        self._bm25 = _BM25([_rag_tokens(t) for t in corpus_txt])
        self.dense_backend = self._init_dense()
        # Real DENSE (embedding) channel, fused into the hybrid ranker below. OFF by
        # default (bm25+tfidf only, reproducing the frozen study); RAG_DENSE=1 enables it.
        self._dense = None
        if os.environ.get("RAG_DENSE"):
            try:
                from case2_flowsheet.core.embeddings import DenseIndex
                di = DenseIndex([f"{d['title']}. {d['text']}" for d in self.docs])
                if di.available:
                    self._dense, self.dense_backend = di, di.backend_name
            except Exception:
                self._dense = None
        # Optional cross-encoder reranker (RAG_RERANK=1): re-ranks the fused top-N more
        # precisely than the fused channels; fails soft to the fused order. Symmetric
        # with case1_reactor.
        self._reranker = None
        self.rerank_backend = "off"
        if os.environ.get("RAG_RERANK"):
            try:
                from case2_flowsheet.core.embeddings import CrossEncoderReranker
                rr = CrossEncoderReranker()
                self.rerank_backend = rr.backend_name
                if rr.available:
                    self._reranker = rr
            except Exception:
                self._reranker = None

    def _init_dense(self):
        """Report the active rationale-retrieval backend. True dense embeddings need an
        offline model (sentence-transformers/torch) that is unavailable here, so the
        channel is a HYBRID lexical retriever: BM25 + TF-IDF fused by reciprocal rank
        fusion (a measured upgrade over raw TF-IDF cosine; see rag_eval.py)."""
        try:
            import sentence_transformers  # noqa: F401  (needs torch; usually absent)
            return "dense(sentence-transformers) + bm25/tfidf hybrid"
        except Exception:
            return "hybrid lexical: bm25 + tfidf (RRF); offline dense embeddings unavailable"

    def _passage(self, d, score=None):
        p = {"id": d["id"], "source_id": d.get("source_id"), "version": d["version"],
             "title": d["title"], "kind": d.get("kind"), "text": d["text"]}
        if "limit" in d:
            p["limit"] = d["limit"]
        if score is not None:
            p["score"] = round(float(score), 4)
        return p

    def _expand_query(self, query):
        """Tokenize + light domain synonym expansion (query side only)."""
        toks = _rag_tokens(query)
        extra = []
        for t in toks:
            extra.extend(_RAG_SYNONYMS.get(t, []))
        return toks + extra

    def _tfidf_scores(self, query):
        return self._cos(self._vec.transform([query]), self._mat).flatten()

    def _rank(self, query, method, _fuse_k=60):
        """Return (order, scores): doc indices best-first under `method` in
        {'tfidf','bm25','hybrid'} (hybrid = RRF of TF-IDF + BM25 [+ dense embeddings when RAG_DENSE=1])."""
        n = len(self.docs)
        tfidf = self._tfidf_scores(query)
        if method == "tfidf":
            return np.argsort(tfidf)[::-1], tfidf
        bm25 = np.array(self._bm25.scores(self._expand_query(query)), float)
        if method == "bm25":
            return np.argsort(bm25)[::-1], bm25
        # hybrid = TWO-LEVEL RRF: fuse the two correlated lexical channels (TF-IDF + BM25)
        # into ONE lexical ranking, THEN fuse lexical + dense, so the dense channel gets
        # ~50% weight instead of being out-voted 2:1 by the redundant lexical rankings.
        # Channel weights are configurable (RAG_W_LEX / RAG_W_DENSE) for weighted RRF.
        def _rrf(orders, weights):
            s = np.zeros(n)
            for o, w in zip(orders, weights):
                rk = np.empty(n, int); rk[o] = np.arange(n)
                s += w / (_fuse_k + rk + 1)
            return s
        lex_rrf = _rrf([np.argsort(tfidf)[::-1], np.argsort(bm25)[::-1]], [1.0, 1.0])
        if self._dense is not None:
            dense_order = np.array([i for i, _s in self._dense.query(query, top_k=n)], int)
            w_lex   = float(os.environ.get("RAG_W_LEX", 1.0))
            w_dense = float(os.environ.get("RAG_W_DENSE", 1.0))
            rrf = _rrf([np.argsort(lex_rrf)[::-1], dense_order], [w_lex, w_dense])
        else:
            rrf = lex_rrf
        return np.argsort(rrf)[::-1], rrf

    def retrieve(self, query, top_k=4, method="hybrid", dedup=True):
        """DENSE rationale channel: HYBRID lexical retrieval — BM25 + TF-IDF cosine fused
        by reciprocal rank fusion (RRF), with light domain query-expansion. By default
        returns DISTINCT source_ids (the highest-ranked chunk per source_id, so duplicate
        spec versions do not clutter the result). `method` in {'hybrid','tfidf','bm25'}.
        (True dense embeddings are offline-unavailable; on this small curated corpus all
        lexical methods are near-ceiling — see rag_eval.py.)"""
        order, sc = self._rank(query, method)
        if self._reranker is not None and len(order):
            pool   = list(order[:max(int(top_k) * 4, 20)])
            scores = self._reranker.rerank(query, [self.docs[i]["text"] for i in pool])
            if scores is not None:
                pool  = [p for _, p in sorted(zip(scores, pool), key=lambda t: -t[0])]
                order = pool + [i for i in order if i not in pool]
        out, seen = [], set()
        for i in order:
            sid = self.docs[i].get("source_id")
            if dedup and sid in seen:
                continue
            seen.add(sid)
            out.append(self._passage(self.docs[i], sc[i]))
            if len(out) >= top_k:
                break
        return out

    def retrieve_constraint(self, query, use_stale=False):
        """
        SPARSE constraint channel: keyword/tag match over 'constraint' chunks.
        Returns the CURRENT (max-version) spec per matched source_id plus a
        version-conflict flag. If use_stale=True (stale-spec scenario), the
        stale version is surfaced instead — to exercise the fail-closed path.
        """
        ql = query.lower()
        hits = [d for d in self.docs if d.get("kind") == "constraint" and
                (any(t in ql for t in d.get("tags", [])) or
                 d["source_id"].lower() in ql or
                 d.get("limit", {}).get("variable", "") in ql)]
        if not hits:   # fall back to dense match restricted to constraints
            hits = [d for d in self.docs if d.get("kind") == "constraint"]
        by_src = defaultdict(list)
        for d in hits:
            by_src[d["source_id"]].append(d)
        out = []
        for src, group in by_src.items():
            group_sorted = sorted(group, key=lambda d: float(d["version"]))
            current = group_sorted[-1]
            chosen  = group_sorted[0] if (use_stale and len(group_sorted) > 1) else current
            p = self._passage(chosen)
            p["version_conflict"] = len(group_sorted) > 1
            p["current_version"]  = current["version"]
            p["is_current"]       = (chosen["version"] == current["version"])
            out.append(p)
        return out

    def current_spec(self, source_id):
        """Return {current, stale[], conflict} for a constraint source_id."""
        group = sorted([d for d in self.docs if d.get("source_id") == source_id],
                       key=lambda d: float(d["version"]))
        if not group:
            return {"current": None, "stale": [], "conflict": False}
        return {"current": self._passage(group[-1]),
                "stale": [self._passage(d) for d in group[:-1]],
                "conflict": len(group) > 1}


INFEAS_BASE = 1e7   # guides DE toward feasibility; NEVER reported as an optimum


class Optimizer:
    """
    Feasibility-aware minimizer behind one interface (Phase 2).

    evaluate_fn(x_dict) -> {objective: float|None, converged: bool,
                            feasible: bool, min_margin: float, raw: dict}
    where min_margin is the minimum NORMALIZED signed constraint margin
    (>=0 feasible). Non-converged points have objective=None and a strongly
    negative min_margin.

    Feasibility-aware handling (non-negotiable #5): non-converged and
    constraint-violating points are EXCLUDED from the optimum — the reported
    optimum is always the best genuinely-feasible EVALUATED point, never a
    penalty value. The DE penalty only GUIDES the search; it is never returned.

    Backends: 'differential_evolution' (robust, dependency-light; default for the
    cheap backend) and 'gp_bo' (sample-efficient feasibility-aware Expected
    Improvement; default for the expensive real-Aspen path).
    """

    def __init__(self, config=CONFIG):
        self.config = config

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _x_dict(arr, variables):
        d = {}
        for v, a in zip(variables, arr):
            d[v["name"]] = int(round(a)) if v["type"] == "int" else float(a)
        return d

    def _record(self, hist, x_dict, res, t0):
        hist.append({"x": x_dict, "objective": res.get("objective"),
                     "feasible": bool(res.get("feasible")),
                     "converged": bool(res.get("converged")),
                     "min_margin": float(res.get("min_margin", -9.99)),
                     "t": round(time.time() - t0, 3)})

    @staticmethod
    def _best_feasible(hist):
        feas = [h for h in hist if h["feasible"] and h["objective"] is not None]
        return min(feas, key=lambda h: h["objective"]) if feas else None

    def optimize(self, problem_spec, evaluate_fn, rng, backend=None):
        variables = problem_spec["variables"]
        backend   = backend or problem_spec.get("backend", "differential_evolution")
        t0, hist  = time.time(), []
        if backend == "gp_bo":
            self._bo(variables, evaluate_fn, rng, hist, t0)
        elif backend == "gp_bo_hardened":
            self._bo_hardened(variables, evaluate_fn, rng, hist, t0)
        else:
            self._de(variables, evaluate_fn, rng, hist, t0)
        best = self._best_feasible(hist)
        return {"x": (best["x"] if best else None),
                "objective": (best["objective"] if best else None),
                "min_margin": (best["min_margin"] if best else None),
                "feasible": best is not None,
                "n_eval": len(hist), "wall_s": round(time.time() - t0, 2),
                "history": hist, "backend": backend,
                "status": "ok" if best else "no_feasible_point_found"}

    # ── differential evolution (feasibility-guided scalar; report best feasible)
    def _de(self, variables, evaluate_fn, rng, hist, t0):
        from scipy.optimize import differential_evolution
        bounds = [(v["lower"], v["upper"]) for v in variables]
        oc = self.config["optimizer"]
        def scalar(arr):
            res = evaluate_fn(self._x_dict(arr, variables))
            self._record(hist, self._x_dict(arr, variables), res, t0)
            if res.get("feasible"):
                return float(res["objective"])
            # infeasible/non-converged: large base + violation (guidance only)
            viol = max(-float(res.get("min_margin", -1.0)), 0.0)
            return INFEAS_BASE + 1e3 * viol
        seed = int(rng.integers(0, 2**31 - 1))
        try:
            differential_evolution(
                scalar, bounds, seed=seed, polish=False,
                maxiter=oc["de_maxiter"], popsize=oc["de_popsize"],
                tol=oc["de_tol"], mutation=(0.5, 1.0), recombination=0.7,
                init="sobol")
        except Exception as err:
            diag_log.error("differential_evolution failed: %s", err)

    # ── feasibility-aware GP Bayesian optimization (Expected Improvement) ──────
    def _bo(self, variables, evaluate_fn, rng, hist, t0):
        """Feasibility-aware GP-BO (constrained EI): an objective GP (converged points) +
        a feasibility GP on the min normalized margin, with local refinement around the
        incumbent so the search can exploit up to the feasibility boundary (where the
        min-duty optimum lives). This is the STUDY optimizer on the real-Aspen path
        (near-optimal there: 898 vs the 887 kW region min). A Sobol+log-EI hardening was
        benchmarked but did not help the real region — see `_bo_hardened`."""
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (Matern, ConstantKernel as C,
                                                       WhiteKernel)
        from sklearn.exceptions import ConvergenceWarning
        from scipy.stats import norm
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lo = np.array([v["lower"] for v in variables], float)
        hi = np.array([v["upper"] for v in variables], float)
        span = np.where(hi > lo, hi - lo, 1.0)
        nrm   = lambda X: (np.atleast_2d(X) - lo) / span     # to [0,1]
        oc = self.config["optimizer"]

        X, Y_obj, Y_feas = [], [], []     # normalized X; objective; feasibility score
        def probe(arr):
            xd  = self._x_dict(arr, variables)
            res = evaluate_fn(xd)
            self._record(hist, xd, res, t0)
            X.append(nrm(arr)[0])
            Y_obj.append(res["objective"] if (res.get("converged") and
                         res.get("objective") is not None) else np.nan)
            Y_feas.append(float(res.get("min_margin", -5.0)))
            return res

        n_init = oc["bo_init"]
        inits  = rng.uniform(lo, hi, size=(n_init, len(variables)))
        inits  = np.vstack([inits, (lo + hi) / 2.0])
        for arr in inits:
            probe(arr)

        for _ in range(oc["bo_iters"]):
            Xa = np.array(X)
            yo = np.array(Y_obj); yf = np.array(Y_feas)
            okm = ~np.isnan(yo)
            if okm.sum() >= 3:
                k_o = C(1.0) * Matern(length_scale=np.ones(len(variables)), nu=2.5) \
                      + WhiteKernel(1e-6, (1e-9, 1e-1))
                gp_o = GaussianProcessRegressor(kernel=k_o, normalize_y=True,
                                                n_restarts_optimizer=2,
                                                random_state=int(rng.integers(1e9)))
                gp_o.fit(Xa[okm], yo[okm])
            else:
                gp_o = None
            k_f = C(1.0) * Matern(length_scale=np.ones(len(variables)), nu=2.5) \
                  + WhiteKernel(1e-4, (1e-7, 1e0))
            gp_f = GaussianProcessRegressor(kernel=k_f, normalize_y=True,
                                            n_restarts_optimizer=2,
                                            random_state=int(rng.integers(1e9)))
            gp_f.fit(Xa, yf)

            best  = self._best_feasible(hist)
            fstar = best["objective"] if best else None
            cand = rng.uniform(lo, hi, size=(400, len(variables)))
            if best is not None:
                bx  = np.array([best["x"][v["name"]] for v in variables], float)
                loc = bx + rng.normal(0, 0.06, size=(120, len(variables))) * span
                cand = np.vstack([cand, np.clip(loc, lo, hi)])
            cn = (cand - lo) / span
            mf, sf = gp_f.predict(cn, return_std=True); sf = np.maximum(sf, 1e-9)
            p_feas  = norm.cdf(mf / sf)                      # P(min_margin >= 0)
            pred_ok = p_feas >= 0.5
            if gp_o is not None and fstar is not None and pred_ok.any():
                mo, so = gp_o.predict(cn, return_std=True); so = np.maximum(so, 1e-9)
                z  = (fstar - mo) / so
                ei = (fstar - mo) * norm.cdf(z) + so * norm.pdf(z)   # min-EI
                acq = np.where(pred_ok, np.maximum(ei, 0.0), -1.0)
            else:
                acq = p_feas                                 # no feasible yet -> seek feasibility
            probe(cand[int(np.argmax(acq))])

    def _bo_hardened(self, variables, evaluate_fn, rng, hist, t0):
        """OPT-IN hardened GP-BO (backend='gp_bo_hardened'): Sobol low-discrepancy initial
        design + local refinement + numerically-stable constrained log-EI, at the same
        eval budget. Benchmarked (bo_benchmark.py) vs the many-eval DE reference: the gap
        improves from ~19.6% to ~12.2% on the THIN-window methanol benchmark and ~1.2% to
        ~0.5% on a synthetic problem. HOWEVER, on the REAL wide-feasible region it does
        NOT help (a nominal check gave 904 vs the original 898 kW), so the STUDY uses the
        plain `_bo`; this is retained for hard thin-feasible problems / as an honest
        benchmark. (An adaptive trust region was tried and dropped — it destabilised the
        thin-window search without improving the mean.)"""
        import warnings
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (Matern, ConstantKernel as C,
                                                       WhiteKernel)
        from sklearn.exceptions import ConvergenceWarning
        from scipy.stats import norm, qmc
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        d = len(variables)
        lo = np.array([v["lower"] for v in variables], float)
        hi = np.array([v["upper"] for v in variables], float)
        span = np.where(hi > lo, hi - lo, 1.0)
        oc = self.config["optimizer"]

        X, Y_obj, Y_feas = [], [], []     # normalized X; objective; feasibility score
        def probe(arr):
            arr = np.clip(arr, lo, hi)
            xd  = self._x_dict(arr, variables)
            res = evaluate_fn(xd)
            self._record(hist, xd, res, t0)
            X.append((arr - lo) / span)
            Y_obj.append(res["objective"] if (res.get("converged") and
                         res.get("objective") is not None) else np.nan)
            Y_feas.append(float(res.get("min_margin", -5.0)))
            return res

        # Sobol initial design (low-discrepancy -> better hits a thin feasible strip than
        # uniform random) + the bounds midpoint.
        n_init = oc["bo_init"]
        sob = qmc.Sobol(d=d, scramble=True, seed=int(rng.integers(1e9)))
        for arr in np.vstack([lo + sob.random(n_init) * span, (lo + hi) / 2.0]):
            probe(arr)

        for _ in range(oc["bo_iters"]):
            Xa = np.array(X); yo = np.array(Y_obj); yf = np.array(Y_feas)
            okm = ~np.isnan(yo)
            gp_o = None
            if okm.sum() >= 3:
                k_o = C(1.0) * Matern(length_scale=np.ones(d), nu=2.5) \
                      + WhiteKernel(1e-6, (1e-9, 1e-1))
                gp_o = GaussianProcessRegressor(kernel=k_o, normalize_y=True,
                                                n_restarts_optimizer=2,
                                                random_state=int(rng.integers(1e9)))
                gp_o.fit(Xa[okm], yo[okm])
            k_f = C(1.0) * Matern(length_scale=np.ones(d), nu=2.5) \
                  + WhiteKernel(1e-4, (1e-7, 1e0))
            gp_f = GaussianProcessRegressor(kernel=k_f, normalize_y=True,
                                            n_restarts_optimizer=2,
                                            random_state=int(rng.integers(1e9)))
            gp_f.fit(Xa, yf)

            best  = self._best_feasible(hist)
            fstar = best["objective"] if best else None
            # candidate pool: global Sobol (explore) + DENSE local sampling inside the
            # trust region around the incumbent (exploit the constraint boundary).
            cg = lo + qmc.Sobol(d=d, scramble=True,
                                seed=int(rng.integers(1e9))).random(256) * span
            cand = cg
            if best is not None:
                bxn = (np.array([best["x"][v["name"]] for v in variables], float) - lo) / span
                loc = np.clip(bxn + rng.normal(0, 0.06, size=(256, d)), 0.0, 1.0)
                cand = np.vstack([cg, lo + loc * span])
            cn = (cand - lo) / span
            mf, sf = gp_f.predict(cn, return_std=True); sf = np.maximum(sf, 1e-9)
            p_feas  = norm.cdf(mf / sf)                      # P(min_margin >= 0)
            pred_ok = p_feas >= 0.5
            if gp_o is not None and fstar is not None and pred_ok.any():
                mo, so = gp_o.predict(cn, return_std=True); so = np.maximum(so, 1e-9)
                z  = (fstar - mo) / so
                ei = (fstar - mo) * norm.cdf(z) + so * norm.pdf(z)   # min-EI
                # CONSTRAINED log-EI: only predicted-feasible candidates compete, so the
                # search exploits right up to the feasibility boundary (where the min-duty
                # optimum lives). log() is numerically stable when EI is tiny there.
                acq = np.where(pred_ok, np.log(np.maximum(ei, 1e-12)), -1e18)
            else:
                acq = p_feas                                 # no feasible yet -> seek feasibility
            probe(cand[int(np.argmax(acq))])


class FlowsheetCopilotAgent:
    """
    LLM planner + RAG + optimizer + fail-closed certificate gate.
    Phase 4 implements the gate machinery (certificate, robustness,
    recommendation-safety, decision-equality, counterfactual, verifier
    consensus). Phase 5 adds the LLM spec compilation; Phase 6 adds run_case
    and the baselines.
    """
    def __init__(self, config=CONFIG, flowsheet=None, rag=None, optimizer=None,
                 registry=None, mock_llm=MOCK_LLM, rng=None, model=None, sim_weak=False):
        self.config      = config
        self.flowsheet   = flowsheet if flowsheet is not None else AspenFlowsheet()
        self.rag         = rag if rag is not None else RAGEngine()
        self.optimizer   = optimizer if optimizer is not None else Optimizer(config)
        self.registry    = registry
        self.mock_llm    = mock_llm
        # Per-agent planner model (cross-model arm, C4): a second agent with a weaker
        # model runs the same arms; every capture records the model that produced it.
        self.model       = model or LLM_MODEL
        # DRY-RUN ONLY: simulate a weaker, more error-prone model in MOCK mode (it invents
        # the distillate threshold on off-nominal cases). OFF by default, so the locked
        # study + the strong-model arms are unaffected. The live run uses a real weaker
        # Claude and this flag stays False.
        self.sim_weak    = sim_weak
        self.rng         = rng if rng is not None else np.random.default_rng(BASE_SEED)
        self.constraints = config["constraints"]
        self.active_vars = [dv for dv in config["decision_vars"] if dv["optimize"]]
        self.results     = []
        # Live-LLM usage meter (R6): accumulates exact token usage across live calls so
        # the one-shot paid run's cost is recorded verbatim. Stays zero in mock mode.
        self._llm_usage  = {"input_tokens": 0, "output_tokens": 0, "n_calls": 0}
        # Contamination meter (live): live plans attempted vs those yielding no usable
        # spec, and plans with >=1 hallucinated citation. Stays zero in mock mode.
        self._llm_plans  = {"total": 0, "contaminated": 0, "hallucinated": 0}
        self._client     = None       # lazy Anthropic client (or an injected fake)
        # On a live resume (fresh process), seed the cost meter from the verbatim log so
        # the spend backstop accounts for spend already incurred before the crash (1.3).
        if not self.mock_llm:
            self._restore_usage_from_log()

    def _backend(self):
        """Optimizer backend by cost regime (R3): the sample-efficient GP-BO on the
        real Aspen path (~0.5 s/run), cheap differential evolution for synthetic test functions.
        Wiring this was the fix for run_case/baselines/counterfactual, which had
        differential_evolution hardcoded — untenable (~1400 evals/opt) on real Aspen."""
        oc = self.config["optimizer"]
        return oc["backend_mock"] if self.flowsheet.mock else oc["backend_real"]

    # ── design evaluation + optimizer evaluate_fn ─────────────────────────────
    def eval_design(self, design, feed_z):
        out = self.flowsheet.evaluate(design, feed_z)
        cm  = constraint_margins(out, self.constraints)
        return {"outputs": out, "converged": out["converged"], **cm}

    def make_evaluate_fn(self, feed_z, constraints=None):
        """Optimizer evaluator. The optimization uses the SPEC's constraints
        (which, in the stale-spec scenario, may be the outdated 0.95 purity), but
        the gate (certify) always verifies against the CURRENT corpus constraints
        — so a design tuned to a stale spec is caught and fails closed."""
        cons = constraints if constraints is not None else self.constraints
        def ev(xd):
            out = self.flowsheet.evaluate(xd, feed_z)
            cm  = constraint_margins(out, cons)
            return {"objective": out.get("reboiler_duty"),
                    "converged": out["converged"], "feasible": cm["feasible"],
                    "min_margin": cm["min_margin"], "raw": out}
        return ev

    # ── LLM planner (decision specification only) ─────────────────────────────
    def proposer_prompt(self):
        """PROPOSER (call 1): compiles the query into a machine-checkable decision
        specification + candidate setpoints + citations. It does NOT write the operator
        narrative — that is the EXPLAINER's job (call 2, post-hoc), so the certificate can
        check the explanation's faithfulness independently of the action (HARMONIZATION
        SPEC §1)."""
        dv_desc = "; ".join(f"{dv['name']} ({dv['type']}, range {dv['lower']}-{dv['upper']})"
                            for dv in self.active_vars)
        nstages = self.config["system"]["column"]["n_stages"]
        return (
            "You are a process-engineering PROPOSER for an Aspen Plus distillation "
            "column. You DO NOT compute or decide the final numbers — a numerical "
            "optimizer and Aspen produce and verify them. Your job is to COMPILE the "
            "user query into a machine-checkable DECISION SPECIFICATION.\n"
            f"The column is FIXED at {nstages} stages — the number of stages is NOT a "
            "decision variable. The ONLY decision variables are EXACTLY these; use these "
            "EXACT names, propose bounds WITHIN these ranges, and do NOT add, rename, or "
            f"remove any: {dv_desc}. The 'recommendation' MUST use these same keys.\n"
            "Return STRICT JSON with keys: 'specs' (a list of 1-3 candidate "
            "specifications), 'recommendation' (a proposed operating point as "
            "{variable: value} using ONLY the keys above — it will be SIMULATED and "
            "verified, not trusted), and 'cited_constraints' (source_ids of the "
            "decision-critical constraints you used, from the CURRENT corpus). Each spec "
            "has: 'objective' ('minimize reboiler_duty'), 'decision_variables' (list of "
            "{name,type,lower_bound,upper_bound} using ONLY the names above), 'constraints' "
            "(list of {name,relation,threshold,source_id,source_version}), 'uncertainty' "
            "(list of {parameter,distribution,range}), and 'required_evidence' "
            "(source_ids). Do NOT invent constraint thresholds; use the retrieved "
            "versioned specs. Do NOT write prose. Emit ONLY the JSON object.")

    # back-compat alias (project_live_cost + older callers referenced system_prompt)
    def system_prompt(self):
        return self.proposer_prompt()

    def explainer_prompt(self):
        """EXPLAINER (call 2): writes the operator narrative for a decision that has
        ALREADY been verified + released. It does NOT choose or change the action — it
        explains WHY the released setpoints meet the specs, citing ONLY retrieved
        knowledge, and RESTATES the exact released setpoints so the certificate can verify
        the narrative is faithful (explanation_consistent + the evidence_ok citation
        check). This is what gives CS2 the narrative-mismatch hallucination type CS1
        measures (HARMONIZATION_SPEC §4)."""
        return (
            "You are a process-engineering EXPLAINER. A distillation design has ALREADY "
            "been chosen and verified by a numerical optimizer + Aspen + a safety "
            "certificate. DO NOT change it. Given the RELEASED setpoints and the retrieved "
            "process knowledge, write a concise operator narrative explaining why this "
            "design meets the product-purity and equipment limits.\n"
            "Return STRICT JSON with keys: 'narrative' (prose for the operator), "
            "'stated_action' (the released operating point you are explaining, as "
            "{reflux_ratio, DtoF, feed_stage} — restate it EXACTLY, do not alter it), and "
            "'cited_constraints' (source_ids you reference, ONLY from the retrieved "
            "knowledge — never invent or cite an un-retrieved source). Emit ONLY the JSON "
            "object.")

    def _client_or_raise(self):
        if getattr(self, "_client", None) is None:
            if not API_KEY:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Export it for the live LLM "
                    "path, or set FCO_MOCK_LLM=1 to plan offline.")
            import anthropic
            self._client = anthropic.Anthropic(api_key=API_KEY)
        return self._client

    def _call_llm(self, user_msg, max_retries=3, backoff=2.0, timeout=60.0,
                  system_prompt=None):
        client, last = self._client_or_raise(), None
        sys_prompt = system_prompt or self.proposer_prompt()
        # HARD spend backstop (fail-safe). Refuse to call at all without a POSITIVE cap,
        # and refuse once the metered cost (incl. spend restored on resume) crosses it.
        if LLM_COST_ABORT_USD <= 0:
            raise RuntimeError(
                "No live spend ceiling set (FCO_COST_ABORT_USD<=0): refusing to call the "
                "API without a hard cap. Export FCO_COST_ABORT_USD (e.g. 2-3).")
        if self.llm_cost_usd() >= LLM_COST_ABORT_USD:
            raise RuntimeError(
                "FCO_COST_ABORT_USD=%.2f reached (metered $%.4f over %d calls); refusing "
                "further live LLM calls (fail-closed on spend)."
                % (LLM_COST_ABORT_USD, self.llm_cost_usd(), self._llm_usage["n_calls"]))
        for attempt in range(max_retries):
            try:
                t_call = time.time()
                resp = client.messages.create(
                    model=self.model, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMP,
                    timeout=timeout, system=sys_prompt,
                    messages=[{"role": "user", "content": user_msg}])
                latency_s = round(time.time() - t_call, 3)
                text  = resp.content[0].text                  # unpack FIRST (1.4)
                model = getattr(resp, "model", self.model)
                u = getattr(resp, "usage", None)              # meter only after a clean unpack
                in_d  = int(getattr(u, "input_tokens", 0) or 0) if u is not None else 0
                out_d = int(getattr(u, "output_tokens", 0) or 0) if u is not None else 0
                if u is not None:
                    self._llm_usage["input_tokens"]  += in_d
                    self._llm_usage["output_tokens"] += out_d
                    self._llm_usage["n_calls"]       += 1
                stop_reason = getattr(resp, "stop_reason", None)
                if stop_reason == "max_tokens":
                    # TRUNCATED: a partial decision spec parses as garbage. Deterministic
                    # (a retry would truncate too) -> fail loudly so the cap is raised.
                    # (usage IS counted above; we paid for the truncated tokens.)
                    raise LLMTruncatedError(
                        f"LLM output truncated at max_tokens={LLM_MAX_TOKENS}; raise "
                        f"LLM_MAX_TOKENS and re-run (response NOT parsed).")
                meta = {"usage_delta": {"input_tokens": in_d, "output_tokens": out_d},
                        "stop_reason": stop_reason, "latency_s": latency_s,
                        "temperature": LLM_TEMP, "max_tokens": LLM_MAX_TOKENS,
                        # journal-grade traceability (request id + cache split; cache fields are
                        # 0/None while no cache_control is used)
                        "request_id": getattr(resp, "_request_id", None),
                        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
                        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
                        "cost_usd_call": round(in_d * LLM_PRICE_IN_PER_TOK +
                                               out_d * LLM_PRICE_OUT_PER_TOK, 6)}
                return text, model, meta
            except LLMTruncatedError:
                raise                                          # non-retryable; do not burn retries
            except Exception as err:
                last = err
                diag_log.warning("LLM attempt %d/%d failed: %s",
                                 attempt + 1, max_retries, err)
                if attempt < max_retries - 1 and backoff > 0:
                    time.sleep(backoff * (2 ** attempt))
        raise RuntimeError(f"LLM planning failed after {max_retries} attempts: {last}")

    def llm_cost_usd(self):
        """Metered live-LLM cost so far (USD) from accumulated EXACT token usage."""
        u = self._llm_usage
        return (u["input_tokens"] * LLM_PRICE_IN_PER_TOK +
                u["output_tokens"] * LLM_PRICE_OUT_PER_TOK)

    def _log_llm_raw(self, case, raw, model, *, system=None, prompt=None,
                     retrieved=None, meta=None, rag_on=None, ground_truth=None):
        """Persist the VERBATIM live-LLM output AND the full, irreplaceable INPUTS
        (one-shot capture; §2 of the campaign blueprint). A paid run cannot be replayed,
        so per live call we append to results/llm_raw_log.jsonl: the full prompt (system
        + user ctx incl. the injected passage text + resolved spec constraints), the
        retrieved set (each passage id/source_id/version/full text/score/rank + the
        query) + the sparse current/stale specs, the raw response + stop_reason, per-call
        + cumulative usage/USD, latency, temperature/max_tokens/model/rag_on, an ISO-UTC
        timestamp, and the ground truth (true spec + stale flag). Logged BEFORE parsing so
        even an unparseable response is preserved. Never invoked in mock mode. The DERIVED
        audit (parsed decision, consensus scores, certificate, verified outputs) is
        persisted per (case,arm) in the result record, joinable on (case_id, system)."""
        from datetime import datetime, timezone
        meta = meta or {}
        rec = {"schema": "llm_raw/2", "ts_utc": datetime.now(timezone.utc).isoformat(),
               "case_id": case.get("id"), "system": system, "model": model,
               "rag_on": rag_on, "temperature": meta.get("temperature"),
               "max_tokens": meta.get("max_tokens"), "stop_reason": meta.get("stop_reason"),
               "latency_s": meta.get("latency_s"),
               "prompt": prompt, "retrieved": retrieved, "raw": raw,
               "usage_call": meta.get("usage_delta"), "cost_usd_call": meta.get("cost_usd_call"),
               "usage_cumulative": dict(self._llm_usage),
               "cost_usd_cumulative": round(self.llm_cost_usd(), 6),
               "ground_truth": ground_truth}
        try:
            with open(os.path.join(RESULT_DIR, "llm_raw_log.jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception as err:
            diag_log.warning("llm_raw_log append failed: %s", err)

    def _log_llm_error(self, case, err):
        """Capture a live-LLM call FAILURE (API error after retries, or truncation)
        verbatim to llm_raw_log.jsonl, so the one-shot record is complete (raw outputs
        + usage + errors). These are run-stopping (resumable), not per-case abstains."""
        rec = {"case_id": case.get("id"), "kind": "llm_error",
               "error": f"{type(err).__name__}: {err}",
               "usage_cumulative": dict(self._llm_usage),
               "cost_usd_cumulative": round(self.llm_cost_usd(), 6)}
        try:
            with open(os.path.join(RESULT_DIR, "llm_raw_log.jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception as e2:
            diag_log.warning("llm_raw_log error-append failed: %s", e2)

    def contamination_report(self):
        """Live-plan contamination = fraction of live plans yielding NO usable spec
        (unparseable / no schema-valid spec). Drives the contamination abort. A gate
        REJECTING a well-formed spec is NOT counted here (it is a reported finding)."""
        p = self._llm_plans
        return {"total_live_plans": p["total"], "contaminated": p["contaminated"],
                "hallucinated": p["hallucinated"],
                "rate": (p["contaminated"] / p["total"]) if p["total"] else 0.0}

    def _restore_usage_from_log(self):
        """Resume safety (1.3): seed the in-memory cost meter from the LAST
        usage_cumulative recorded in llm_raw_log.jsonl, so a crash+resume in a fresh
        process does NOT reset the FCO_COST_ABORT_USD backstop to $0. Over-counting
        (e.g. a stale log from a prior run) fails SAFE — it can only refuse earlier,
        never overspend. Delete llm_raw_log.jsonl for a genuinely clean run."""
        p = os.path.join(RESULT_DIR, "llm_raw_log.jsonl")
        if not os.path.exists(p):
            return
        last = None
        try:
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                u = json.loads(line).get("usage_cumulative")
                if u:
                    last = u
        except Exception as err:
            diag_log.warning("could not restore usage from llm_raw_log: %s", err)
            return
        if last:
            self._llm_usage = {"input_tokens": int(last.get("input_tokens", 0)),
                               "output_tokens": int(last.get("output_tokens", 0)),
                               "n_calls": int(last.get("n_calls", 0))}
            diag_log.info("resume: restored live usage %s ($%.4f) from llm_raw_log",
                          self._llm_usage, self.llm_cost_usd())

    def project_live_cost(self, cases, systems=None):
        """R6 cost PROJECTION: estimate the USD cost of a LIVE run over (cases x
        systems) by building the EXACT user message each plan() would send and pricing
        the estimated tokens. Makes ZERO API calls — call this (or live_cost.py) before
        approving a paid run. B2 calls no LLM; every other system calls plan() once per
        case. Output tokens are charged at the worst-case max_tokens cap."""
        systems = systems or (["B0_ungated_llm", "B1_llm_rag_nogate", "full"] + ABLATIONS)
        llm_systems = [s for s in systems if s != "B2_optimizer_gate"]
        # GATED arms (full + ablations except ablate_gate) ALSO call the EXPLAINER (call 2)
        # on release; count it as a worst case (every gated arm releases, every case).
        gated_expl = [s for s in llm_systems
                      if s in ({"full"} | (set(ABLATIONS) - {"ablate_gate"}))]
        sys_toks  = _estimate_tokens(self.proposer_prompt())
        expl_toks = _estimate_tokens(self.explainer_prompt())
        n_calls, n_expl, in_toks = 0, 0, 0
        for case in cases:
            spec_constraints = self._resolve_constraints(
                stale_sources=self._case_stale_sources(case),
                drop_sources=self._case_drop_sources(case))
            rationale = self.rag.retrieve(
                "minimize reboiler duty meeting purity specs within equipment limits", top_k=4)
            ctx = json.dumps({"case": case, "constraints": spec_constraints,
                              "rationale_passages": rationale}, default=str)
            per_call_in = sys_toks + _estimate_tokens(f"Compile a decision specification "
                                                      f"for this case.\n{ctx}")
            for _ in llm_systems:
                n_calls += 1
                in_toks += per_call_in
            expl_ctx = json.dumps({"rationale_passages": rationale,
                                   "released_setpoints": {}}, default=str)
            expl_in = expl_toks + _estimate_tokens(
                f"Explain this RELEASED, verified distillation design for the operator.\n{expl_ctx}")
            for _ in gated_expl:                             # worst case: every gated arm releases
                n_calls += 1
                n_expl += 1
                in_toks += expl_in
        out_toks = n_calls * LLM_MAX_TOKENS                  # worst case: every call maxes out
        usd = in_toks * LLM_PRICE_IN_PER_TOK + out_toks * LLM_PRICE_OUT_PER_TOK
        return {"model": self.model, "n_live_calls": n_calls,
                "n_proposer_calls": n_calls - n_expl,
                "n_explainer_calls_worstcase": n_expl,
                "llm_systems_per_case": llm_systems, "n_cases": len(cases),
                "est_input_tokens": int(in_toks),
                "worstcase_output_tokens": int(out_toks),
                "price_in_per_MTok": LLM_PRICE_IN_PER_TOK * 1e6,
                "price_out_per_MTok": LLM_PRICE_OUT_PER_TOK * 1e6,
                "projected_worstcase_usd": round(usd, 3),
                "note": "OFFLINE worst-case (~4 chars/token; output at the max_tokens cap; "
                        "every gated arm assumed to release -> +1 explainer call). No API "
                        "call made. An approved live run meters exact usage."}

    # back-compat: stale_spec=True -> the original distillate+bottoms purity pair (the
    # only stale-versioned specs in the locked study); stale_sources lists more.
    _STALE_DEFAULT = ("SPEC-DIST-PURITY", "SPEC-BOT-PURITY")

    def _case_stale_sources(self, case):
        """The set of source_ids whose STALE version this case uses. Reproduces the
        locked C_stale_spec exactly (stale_spec=True -> the purity pair) while letting
        new cases target a stale flooding/duty limit or a partial mix (a distribution)."""
        s = set(case.get("stale_sources") or ())
        if case.get("stale_spec"):
            s |= set(self._STALE_DEFAULT)
        return s

    def _case_drop_sources(self, case):
        """source_ids this case removes from the corpus (out-of-corpus / missing-constraint trap).
        The gate still owns the true limit, so a design violating a dropped constraint fails closed."""
        return set(case.get("drop_sources") or ())

    def _resolve_constraints(self, stale_sources=None, drop_sources=None):
        """Build the spec's constraint list from the corpus. For a source_id in
        `stale_sources` the OUTDATED version is used (the stale-spec trap); all others
        use the CURRENT version. `drop_sources` source_ids are OMITTED entirely (the
        out-of-corpus / missing-constraint trap): the model cannot ground them and must
        abstain or fabricate. The GATE always verifies against the CURRENT CONFIG
        constraints regardless, so a stale- or gap-affected design is caught and fails closed."""
        stale_sources = stale_sources or set()
        drop_sources = drop_sources or set()
        out = []
        for c in self.constraints:
            if c["source_id"] in drop_sources:
                continue                                    # out-of-corpus: not retrievable
            cs  = self.rag.current_spec(c["source_id"])
            doc = cs["current"]
            if c["source_id"] in stale_sources and cs["stale"]:
                doc = cs["stale"][-1]                       # outdated spec for THIS source
            thr = doc["limit"]["value"] if (doc and "limit" in doc) else c["threshold"]
            out.append({"name": c["name"], "relation": c["relation"],
                        "threshold": float(thr), "scale": c["scale"],
                        "critical": c["critical"], "source_id": c["source_id"],
                        "source_version": doc["version"] if doc else c["source_version"]})
        return out

    def _normalize_spec_constraints(self, constraints):
        """A live LLM emits spec constraints as {name, relation, threshold, source_id,
        source_version} -- WITHOUT the AUTHOR-owned 'scale' (and may name them differently)
        that constraint_margins needs (else KeyError: 'scale'). Match each to the CONFIG
        constraint (by NAME first -- so the mock, whose names already match, is a NO-OP and
        result-neutral; then by source_id) and take CONFIG's name(=output key)/relation/
        scale/critical, while KEEPING the LLM's threshold (which may be the stale value --
        the trap) and source_version. Un-mappable constraints are dropped; the gate still
        verifies against CONFIG, so the optimizer search is only normalized, never trusted."""
        by_name = {c["name"]: c for c in self.constraints}
        by_sid  = {c["source_id"]: c for c in self.constraints}
        out = []
        for c in (constraints or []):
            base = by_name.get(c.get("name")) or by_sid.get(c.get("source_id"))
            if base is None:
                continue
            out.append({"name": base["name"], "relation": base["relation"],
                        "threshold": float(c.get("threshold", base["threshold"])),
                        "scale": base["scale"], "critical": base["critical"],
                        "source_id": base["source_id"],
                        "source_version": c.get("source_version", base["source_version"])})
        return out

    # aliases a real LLM tends to use for the study's decision keys
    _DV_ALIAS = {"distillate_to_feed_ratio": "DtoF", "distillate_to_feed": "DtoF",
                 "d_to_f": "DtoF", "dtof": "DtoF", "reflux": "reflux_ratio",
                 "rr": "reflux_ratio", "reflux_rate": "reflux_ratio",
                 "feed_location": "feed_stage", "feed_tray": "feed_stage"}

    def _normalize_spec_decision_vars(self, dvs):
        """The flowsheet supports a FIXED set of decision variables; the LLM must search
        within that interface. Map the LLM's decision_variables to the supported optimized
        ones (by name + aliases), CLAMP bounds into the CONFIG ranges, DROP unknowns (e.g.
        number_of_stages -- the column has a fixed stage count), and ensure every optimized
        variable is present (fill from CONFIG if omitted). Mock dvs already match -> no-op."""
        cfg_by_name = {dv["name"]: dv for dv in self.config["decision_vars"]}
        active = [dv for dv in self.active_vars]
        out, seen = [], set()
        for dv in (dvs or []):
            name = self._DV_ALIAS.get(str(dv.get("name", "")).lower(), dv.get("name"))
            base = cfg_by_name.get(name)
            if base is None or not base.get("optimize") or name in seen:
                continue                                 # unknown / non-optimized / dup
            try:
                lo = max(float(dv.get("lower_bound", base["lower"])), base["lower"])
                hi = min(float(dv.get("upper_bound", base["upper"])), base["upper"])
            except (TypeError, ValueError):
                lo, hi = base["lower"], base["upper"]
            if lo >= hi:
                lo, hi = base["lower"], base["upper"]
            out.append({"name": name, "type": base["type"], "lower_bound": lo, "upper_bound": hi})
            seen.add(name)
        for base in active:                              # ensure all optimized vars present
            if base["name"] not in seen:
                out.append({"name": base["name"], "type": base["type"],
                            "lower_bound": base["lower"], "upper_bound": base["upper"]})
        return out

    def _normalize_recommendation(self, rec):
        """Map the LLM recommendation to the flowsheet's design keys (aliases), keep ONLY
        supported keys, CLAMP to the CONFIG ranges, and fill any missing optimized key with
        the range midpoint, so eval_design(rec) always works. The trap value (e.g. a low
        reflux) is preserved (it is in range); only out-of-interface keys/values are fixed."""
        if not isinstance(rec, dict):
            return None
        cfg_by_name = {dv["name"]: dv for dv in self.config["decision_vars"]}
        out = {}
        for k, v in rec.items():
            name = self._DV_ALIAS.get(str(k).lower(), k)
            base = cfg_by_name.get(name)
            if base is None:
                continue
            try:
                val = min(max(float(v), base["lower"]), base["upper"])
            except (TypeError, ValueError):
                continue
            out[name] = int(round(val)) if base["type"] == "int" else val
        for dv in self.active_vars:
            if dv["name"] not in out:
                mid = (dv["lower"] + dv["upper"]) / 2.0
                out[dv["name"]] = int(round(mid)) if dv["type"] == "int" else mid
        return out

    def _sparse_capture(self, stale_sources):
        """The sparse constraint channel's resolved current/stale specs per source_id,
        for the one-shot capture (the audit shows EXACTLY which version each
        decision-critical spec used)."""
        out = {}
        for c in self.constraints:
            cs = self.rag.current_spec(c["source_id"])
            out[c["source_id"]] = {
                "current": cs.get("current"), "stale": cs.get("stale"),
                "used": ("stale" if (c["source_id"] in stale_sources and cs.get("stale"))
                         else "current")}
        return out

    @staticmethod
    def validate_spec(spec):
        """Schema-validate one decision spec; return (ok, errors). Malformed
        specs are REJECTED, never silently coerced."""
        errs = []
        if not isinstance(spec, dict):
            return False, ["spec is not an object"]
        if "reboiler_duty" not in str(spec.get("objective", "")):
            errs.append("objective must minimize reboiler_duty")
        dvs = spec.get("decision_variables")
        if not isinstance(dvs, list) or not dvs:
            errs.append("decision_variables missing/empty")
        else:
            for dv in dvs:
                if not all(k in dv for k in ("name", "type", "lower_bound", "upper_bound")):
                    errs.append(f"decision_variable missing fields: {dv}")
                elif dv["lower_bound"] >= dv["upper_bound"]:
                    errs.append(f"bad bounds for {dv.get('name')}")
        cons = spec.get("constraints")
        if not isinstance(cons, list) or not cons:
            errs.append("constraints missing/empty")
        else:
            for c in cons:
                if not all(k in c for k in ("name", "relation", "threshold",
                                            "source_id", "source_version")):
                    errs.append(f"constraint missing fields: {c}")
                elif c["relation"] not in (">=", "<="):
                    errs.append(f"bad relation: {c.get('relation')}")
        return (len(errs) == 0), errs

    def citation_faithfulness(self, spec, rel_tol=1e-6):
        """Anti-hallucination check (on-thesis): every constraint threshold the LLM
        STATES must match a REAL corpus value for that source_id (the current OR a stale
        version). A value matching NO version is a hallucinated (invented) number -> the
        spec is rejected. Matching a STALE value is NOT hallucination (that is the
        stale-spec trap, caught later by current-spec re-verification). The mock never
        hallucinates (it builds thresholds from the corpus), so this is neutral in mock
        and load-bearing only on the live path. Returns (ok, mismatches)."""
        mism = []
        for c in spec.get("constraints", []):
            sid, thr = c.get("source_id"), c.get("threshold")
            cs = self.rag.current_spec(sid)
            if cs.get("current") is None:
                continue                                   # non-corpus source_id -> skip
            vals = [float(doc["limit"]["value"]) for doc in [cs["current"], *cs["stale"]]
                    if doc and "limit" in doc]
            if not vals:
                continue
            if thr is None or not any(abs(float(thr) - v) <= rel_tol * max(1.0, abs(v))
                                      for v in vals):
                mism.append({"source_id": sid, "cited": thr, "corpus_values": vals})
        return (len(mism) == 0, mism)

    def _spec_to_problem(self, spec, backend=None):
        """Translate a validated spec into the optimizer problem_spec."""
        return {"variables": [{"name": dv["name"], "type": dv["type"],
                               "lower": dv["lower_bound"], "upper": dv["upper_bound"]}
                              for dv in spec["decision_variables"]],
                "backend": backend}

    def _mock_decision(self, case, spec_constraints, rag_on=True):
        """Fixed, well-formed decision object (FCO_MOCK_LLM). Emits consensus_K
        candidate specs differing in bounds/initialization, a proposed operating
        point (verified, not trusted), citations, and a rationale.

        rag_on=True is the GROUNDED plan (thresholds copied from the retrieved corpus
        specs -> the citation gate passes), unchanged from the locked study. rag_on=False
        SIMULATES an ungrounded LLM: with no retrieved specs it cannot ground the exact
        current threshold and states an INVENTED distillate-purity value (0.98 -- matching
        no corpus version), so the citation gate flags the hallucination. This makes the
        measured RAG ablation (P2) exercise the hallucination RAG prevents even offline."""
        K = self.config["optimizer"]["consensus_K"]
        base_dvs = [{"name": dv["name"], "type": dv["type"],
                     "lower_bound": dv["lower"], "upper_bound": dv["upper"]}
                    for dv in self.active_vars]
        unc = [{"parameter": "z_methanol", "distribution": "uniform",
                "range": [case["feed_z"] - self.config["robustness"]["delta"],
                          case["feed_z"] + self.config["robustness"]["delta"]]}]
        req = [c["source_id"] for c in self.constraints if c["critical"]]
        used_constraints = [dict(c) for c in spec_constraints]
        if not rag_on:                           # ungrounded: invent the purity threshold
            for c in used_constraints:
                if c["source_id"] == "SPEC-DIST-PURITY":
                    c["threshold"] = 0.98        # INVENTED (corpus has only 0.99 / 0.95)
        if self.sim_weak and case.get("id") != "C0_nominal":
            # DRY-RUN ONLY: a weaker model invents the threshold more often (caught by the
            # citation gate for every arm). OFF by default; live run uses a real model.
            for c in used_constraints:
                if c["source_id"] == "SPEC-DIST-PURITY":
                    c["threshold"] = 0.97        # INVENTED (matches no corpus version)
        specs = []
        for k in range(K):                       # candidates differ in RR lower bound
            dvs = [dict(dv) for dv in base_dvs]
            for dv in dvs:
                if dv["name"] == "reflux_ratio":
                    dv["lower_bound"] = [0.5, 1.5, 2.0][k % 3]
            specs.append({"objective": "minimize reboiler_duty",
                          "decision_variables": dvs,
                          "constraints": [dict(c) for c in used_constraints],
                          "uncertainty": unc, "required_evidence": req})
        # proposed operating point: a sensible guess. In the stale-spec case the
        # LLM is misled and proposes a 0.95-tuned (low-reflux) point that VIOLATES
        # the true current 0.99 spec -> recommendation_safe will reject it.
        # proposed point: D:F tracks the feed (a competent plan); reflux is a
        # robust-ish guess for the current spec, but in the stale-spec case the
        # LLM is misled to a LOW reflux tuned to the 0.95/0.03 pair, which
        # VIOLATES the true current 0.99/0.01 -> recommendation_safe rejects it.
        df0 = round(case["feed_z"], 2)
        # the LLM is misled to a LOW-reflux (0.95/0.03-tuned) point ONLY when a PURITY spec
        # is stale (the binding constraint -> the point violates the current 0.99/0.01).
        # A stale flooding/duty limit is SLACK at the min-duty optimum, so the plan stays
        # the safe one (the gate then correctly RELEASES it -> a precision check, not a trap).
        purity_stale = bool(self._case_stale_sources(case) &
                            {"SPEC-DIST-PURITY", "SPEC-BOT-PURITY"})
        rec = ({"reflux_ratio": 1.9, "DtoF": df0, "feed_stage": 13}
               if purity_stale else
               {"reflux_ratio": 2.7, "DtoF": df0, "feed_stage": 13})
        # PROPOSER emits NO prose (the explainer writes the operator narrative, call 2).
        return {"specs": specs, "recommendation": rec,
                "cited_constraints": req, "model": "mock"}

    def plan(self, case, system="full", rag_on=True):
        """Retrieve knowledge, then produce + validate the decision object.

        `rag_on=False` is the MEASURED RAG ablation (P2): the retrieved context (the
        rationale passages AND the resolved current/stale spec constraints) is STRIPPED
        from the prompt, so the LLM must state thresholds ungrounded -> the citation gate
        measures the hallucination RAG prevents. The gate itself is unchanged. `system`
        is the arm label, recorded in the one-shot capture (joins the result record)."""
        stale_sources    = self._case_stale_sources(case)
        spec_constraints = self._resolve_constraints(stale_sources=stale_sources,
                                                     drop_sources=self._case_drop_sources(case))
        query = "minimize reboiler duty meeting purity specs within equipment limits"
        rationale_passages = self.rag.retrieve(query, top_k=4)
        # what the prompt actually carries (stripped for the measured RAG ablation)
        prompt_passages    = rationale_passages if rag_on else []
        prompt_constraints = spec_constraints   if rag_on else []
        ranked_passages    = [{**p, "rank": i} for i, p in enumerate(prompt_passages)]
        ground_truth = {"feed_z": case.get("feed_z"),
                        "stale_sources": sorted(stale_sources),
                        "true_constraints": self._resolve_constraints(stale_sources=set())}
        capture = {"system": system, "rag_on": rag_on, "model": None, "meta": None,
                   "prompt": None, "retrieved": None, "hallucinated": 0,
                   "parsed_decision": None, "candidate_specs": [],
                   "ground_truth": ground_truth}
        if self.mock_llm:
            decision = self._mock_decision(case, spec_constraints, rag_on=rag_on)
            model = "mock"
            capture["model"] = "mock"
        else:
            tool_log, _tooluse_spec = None, None
            # Tool-use grounds the GROUNDED arms only; the no_rag (bare) arm (rag_on=False)
            # must stay one-shot with no tools, or "bare" isn't bare in the three-stage matrix.
            if (os.environ.get("FCO_TOOL_USE") and rag_on) or os.environ.get("FCO_MCP_ONLY"):
                # GROUNDED TOOL-USE proposer: the LLM calls the real MCP tools to ground its
                # spec. Default = retrieve_constraints / retrieve_knowledge / simulate_design.
                # FCO_MCP_ONLY (+MCP-only ablation cell) exposes simulate_design ONLY (no
                # retrieval tools), isolating MCP's standalone contribution. Cost is metered
                # PER CALL through a thin wrapper that reuses the same $ ceiling as _call_llm.
                from types import SimpleNamespace
                from case2_flowsheet import mcp_server
                _mcp_only = bool(os.environ.get("FCO_MCP_ONLY"))
                _agent, _real = self, self._client_or_raise()

                _tooluse_calls = []                 # per-call telemetry for the meta record

                def _metered_create(**kw):
                    if _agent.llm_cost_usd() >= LLM_COST_ABORT_USD:
                        err = RuntimeError(
                            "FCO_COST_ABORT_USD=%.2f reached during tool-use (metered $%.4f)"
                            % (LLM_COST_ABORT_USD, _agent.llm_cost_usd()))
                        _agent._log_llm_error(case, err)   # log before failing closed on spend
                        raise err
                    _t0 = time.time()
                    r = _real.messages.create(**kw)
                    u = getattr(r, "usage", None)
                    if u is not None:
                        _agent._llm_usage["input_tokens"]  += int(getattr(u, "input_tokens", 0) or 0)
                        _agent._llm_usage["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)
                        _agent._llm_usage["n_calls"]       += 1
                    # per-call telemetry (journal-grade): request id, tokens, cache split, latency,
                    # stop_reason -- recorded into the raw-log meta so each LOOP call is traceable.
                    _tooluse_calls.append({
                        "request_id": getattr(r, "_request_id", None),
                        "input_tokens": getattr(u, "input_tokens", None),
                        "output_tokens": getattr(u, "output_tokens", None),
                        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
                        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
                        "stop_reason": getattr(r, "stop_reason", None),
                        "latency_s": round(time.time() - _t0, 3)})
                    return r

                metered = SimpleNamespace(messages=SimpleNamespace(create=_metered_create))
                spec, tool_log, raw = mcp_server.tooluse_decision_spec(
                    metered, self.model, self.flowsheet, self.rag, case.get("feed_z"),
                    mcp_only=_mcp_only)
                # raw = the model's VERBATIM text (the capture source of truth; a parse failure
                # stays diagnosable); the parsed spec is consumed as `decision` below.
                _tooluse_spec, model = spec, self.model
                meta = {"tool_use": True, "mcp_only": _mcp_only,
                        "tool_calls": [e["tool"] for e in tool_log],
                        "llm_calls": _tooluse_calls,          # per-call telemetry (loop-traceable)
                        "cost_usd_cumulative": round(self.llm_cost_usd(), 6)}
                user_msg = "[tool-use grounded proposer]"
            else:
                ctx = json.dumps({"case": case, "constraints": prompt_constraints,
                                  "rationale_passages": ranked_passages}, default=str)
                user_msg = f"Compile a decision specification for this case.\n{ctx}"
                try:
                    raw, model, meta = self._call_llm(user_msg)
                except Exception as err:
                    # API failure (after retries) or truncation: capture verbatim, then
                    # PROPAGATE to stop the run (resumable). NOT a per-case abstain and NOT
                    # contamination — no usable response was received.
                    self._log_llm_error(case, err)
                    raise
            prompt_rec    = {"system": self.system_prompt(), "user": user_msg}
            retrieved_rec = {"query": query, "rag_on": rag_on,
                             "rationale_passages": ranked_passages,
                             "sparse_specs": self._sparse_capture(stale_sources),
                             "tool_log": tool_log}
            # verbatim one-shot capture (log BEFORE parse -> even garbage is preserved)
            self._log_llm_raw(case, raw, model, system=system, prompt=prompt_rec,
                              retrieved=retrieved_rec, meta=meta, rag_on=rag_on,
                              ground_truth=ground_truth)
            self._llm_plans["total"] += 1
            capture.update({"model": model, "meta": meta, "prompt": prompt_rec,
                            "retrieved": retrieved_rec})
            try:
                # tool-use: consume the spec already parsed (extract_last_json) by
                # tooluse_decision_spec -- raw is now verbatim prose+JSON, so the greedy
                # one-shot regex must not re-parse it. An empty spec flows on as {} to the
                # downstream validation exactly as before. One-shot path: unchanged.
                if tool_log is not None:
                    decision = _tooluse_spec or {}
                else:
                    decision = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            except Exception as err:
                diag_log.error("LLM spec parse failed: %s", err)
                self._llm_plans["contaminated"] += 1       # unparseable = contamination
                return {"ok": False, "error": f"unparseable LLM output: {err}",
                        "model": model, "contamination": True, "system": system,
                        "rag_on": rag_on, "capture": capture, "ground_truth": ground_truth}
        # map the recommendation to the tool's design keys + clamp to range, so
        # eval_design(rec)/recommendation_safe always work on real LLM output (no-op for
        # the mock, whose rec already uses {reflux_ratio, DtoF, feed_stage} in range).
        if isinstance(decision, dict) and decision.get("recommendation") is not None:
            decision["recommendation"] = self._normalize_recommendation(decision["recommendation"])
        # validate every candidate spec; reject malformed output AND hallucinated
        # citations (a stated threshold matching no corpus value).
        valid, bad, halluc = [], [], 0
        for i, s in enumerate(decision.get("specs", [])):
            ok, errs = self.validate_spec(s)
            if ok:
                faithful, mism = self.citation_faithfulness(s)
                if not faithful:
                    ok, errs, halluc = False, ["hallucinated citation: %s" % mism], halluc + 1
            (valid if ok else bad).append((i, s, errs))
        if halluc and not self.mock_llm:
            self._llm_plans["hallucinated"] += 1
        capture["hallucinated"]    = halluc
        capture["candidate_specs"] = decision.get("specs", [])
        capture["parsed_decision"] = {"recommendation": decision.get("recommendation"),
                                      "cited_constraints": decision.get("cited_constraints"),
                                      "rationale": decision.get("rationale")}
        # normalize each VALID spec to the tool interface so the optimizer works on real
        # LLM output (mock is a no-op): attach the AUTHOR-owned scale/name to constraints,
        # and map/clamp/complete the decision variables to the supported fixed set.
        for _i, _s, _e in valid:
            _s["constraints"] = self._normalize_spec_constraints(_s.get("constraints"))
            _s["decision_variables"] = self._normalize_spec_decision_vars(
                _s.get("decision_variables"))
        if not valid:
            # Ungrounded vs contaminated (A8, observed live in the B_mcp cell): a spec that is
            # structurally sound but whose EVERY cited source lies outside the corpus (the
            # unknown-source filter emptied its constraints) is the no-retrieval cell's measured
            # fabrication phenomenon — the system abstains fail-closed (reported finding).
            # Contamination stays reserved for capture garbage (unparseable / structurally bad),
            # per its own definition above ("DISTINCT from the gate rejecting a well-formed spec").
            ungrounded = any(isinstance(s, dict) and s.get("_dropped_unknown_sources")
                             and set(errs) == {"constraints missing/empty"}
                             for _i, s, errs in bad)
            diag_log.error("No usable candidate spec (%s): %s",
                           "ungrounded sources" if ungrounded else "malformed/hallucinated", bad)
            if not self.mock_llm:
                if ungrounded:
                    if not halluc:                         # avoid double count with line above
                        self._llm_plans["hallucinated"] += 1
                else:
                    self._llm_plans["contaminated"] += 1   # JSON but no usable spec
            return {"ok": False,
                    "error": ("no grounded constraints: every cited source is outside the "
                              "corpus (fail-closed ungrounded abstention)" if ungrounded
                              else "no schema-valid/faithful specification"),
                    "bad": bad, "model": model,
                    "contamination": (not self.mock_llm) and (not ungrounded),
                    "ungrounded": ungrounded,
                    "system": system, "rag_on": rag_on, "hallucinated": halluc,
                    "capture": capture, "ground_truth": ground_truth}
        return {"ok": True, "decision": decision,
                "specs": [s for _, s, _ in valid],
                "spec_constraints": spec_constraints,
                "rationale_passages": rationale_passages, "model": model,
                "system": system, "rag_on": rag_on, "hallucinated": halluc,
                "capture": capture, "ground_truth": ground_truth}

    # ── EXPLAINER (call 2): post-hoc operator narrative for a verified design ──────
    def _mock_explain(self, released_x, cited):
        """Faithful canned explainer: restates the released action, cites the (retrieved)
        critical constraints, writes a short narrative. Keeps the mock path result-neutral
        (explanation_consistent + the evidence_ok citation check stay True)."""
        rr, df = released_x.get("reflux_ratio"), released_x.get("DtoF")
        fs = released_x.get("feed_stage")
        narrative = ((f"Operate at reflux ratio {float(rr):.2f}, distillate-to-feed "
                      f"{float(df):.2f}, feed stage {int(round(float(fs)))}: this meets the "
                      f"distillate (>=0.99) and bottoms (<=0.01) methanol specs within the "
                      f"flooding and reboiler-duty limits, near minimum reboiler duty.")
                     if rr is not None else
                     "The released design meets the current product-purity and equipment limits.")
        return {"narrative": narrative,
                "stated_action": {"reflux_ratio": rr, "DtoF": df, "feed_stage": fs},
                "cited_constraints": list(cited or []), "model": "mock"}

    def explain(self, case, released_x, decision, rationale_passages, spec_constraints,
                system="full"):
        """EXPLAINER (call 2, post-hoc): given the RELEASED + verified design, write the
        operator narrative. Returns {narrative, stated_action, cited_constraints, model}.
        It does NOT choose the action. The mock explainer is FAITHFUL (so the gated
        decision is unchanged); only a live, UNFAITHFUL narrative (wrong action or a
        fabricated/un-retrieved citation) trips explanation_consistent / evidence_ok."""
        cited = decision.get("cited_constraints", [])
        if self.mock_llm:
            return self._mock_explain(released_x, cited)
        ranked = [{**p, "rank": i} for i, p in enumerate(rationale_passages or [])]
        ctx = json.dumps({"case": case, "released_setpoints": released_x,
                          "cited_constraints": cited, "rationale_passages": ranked},
                         default=str)
        user_msg = "Explain this RELEASED, verified distillation design for the operator.\n" + ctx
        raw, model, meta = self._call_llm(user_msg, system_prompt=self.explainer_prompt())
        self._log_llm_raw(case, raw, model, system=f"{system}:explainer",
                          prompt={"system": self.explainer_prompt(), "user": user_msg},
                          retrieved={"rationale_passages": ranked,
                                     "released_setpoints": released_x},
                          meta=meta, rag_on=True,
                          ground_truth={"released_setpoints": released_x})
        try:
            d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        except Exception as err:
            diag_log.error("explainer parse failed: %s", err)
            return {"narrative": "", "stated_action": None, "cited_constraints": [],
                    "model": model, "parse_failed": True}
        return {"narrative": d.get("narrative", ""), "stated_action": d.get("stated_action"),
                "cited_constraints": d.get("cited_constraints", []), "model": model}

    def _explainer_retrieved_ids(self, rationale_passages, spec_constraints):
        """source_ids the explainer is ALLOWED to cite = those it was SHOWN (the dense
        rationale passages + the sparse resolved spec constraints). A citation outside this
        set is a fabricated / un-retrieved citation."""
        ids = {p.get("source_id") for p in (rationale_passages or [])}
        ids |= {c.get("source_id") for c in (spec_constraints or [])}
        return {i for i in ids if i}

    def _stated_action_matches(self, stated, released):
        """The explainer's stated action must equal the released setpoints (same tols as
        decision_equals_verified). A mismatch is the narrative-mismatch hallucination.
        Alias-maps the stated keys (a real explainer may rename DtoF etc.) but does NOT
        fill missing keys -- an omitted setpoint is genuinely unfaithful."""
        if not stated or not released:
            return False
        s = {self._DV_ALIAS.get(str(k).lower(), k): v for k, v in stated.items()}
        g = self.config["gate"]
        for dv in self.active_vars:
            n = dv["name"]
            if n not in s or n not in released:
                return False
            try:
                if abs(float(s[n]) - float(released[n])) > g["rec_abs_tol"].get(n, 1e-6):
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _augment_cert_with_explainer(self, cert, explainer, released_x, retrieved_ids):
        """Fold the explainer's faithfulness into the cert (post-hoc; NO Aspen re-run):
        evidence_ok ALSO requires every explainer citation to be a retrieved source;
        explanation_consistent requires the explainer's stated action == the released
        design. Recomputes the fail-closed reason."""
        expl_cited = set((explainer or {}).get("cited_constraints") or [])
        cited_ok = expl_cited.issubset(set(retrieved_ids or []))
        consistent = self._stated_action_matches((explainer or {}).get("stated_action"), released_x)
        cert.evidence_ok = bool(cert.evidence_ok and cited_ok)
        cert.explanation_consistent = bool(consistent)
        cert.metrics["explainer_cited_ok"] = cited_ok
        cert.metrics["explanation_consistent"] = consistent
        cert.metrics["explainer_cited"] = sorted(expl_cited)
        if passes(cert):
            cert.reason = "PASS"
        else:
            fails = [n for n, v in (("converged", cert.converged), ("margins_ok", cert.margins_ok),
                                    ("robust", cert.robust), ("evidence_ok", cert.evidence_ok),
                                    ("recommendation_safe", cert.recommendation_safe),
                                    ("decision_equals_verified", cert.decision_equals_verified),
                                    ("explanation_consistent", cert.explanation_consistent))
                     if not v]
            cert.reason = "FAIL-CLOSED: " + ", ".join(fails)
        return cert

    # ── certificate components ────────────────────────────────────────────────
    def robustness(self, design, feed_z):
        """MONTE-CARLO robustness over the feed-composition uncertainty band, unified in
        FORM with case1_reactor (feasible fraction over an ensemble must reach r_min). The
        disturbance source differs by problem (CS1: EnKF posterior; CS2: a +/-delta feed band)
        but both draw an N-member ensemble and require >= r_min feasible. Seeded -> deterministic."""
        rb = self.config["robustness"]; d, n = rb["delta"], rb["n_samples"]
        rng = np.random.default_rng(rb.get("seed", 0))
        zs = rng.uniform(feed_z - d, feed_z + d, size=n)   # MC sample over the uncertainty band
        feas, worst = 0, 1e9
        for z in zs:
            r = self.eval_design(design, float(z))
            feas += int(r["feasible"]); worst = min(worst, r["min_margin"])
        return {"feasible_fraction": feas / n, "worst_min_margin": float(worst), "n": int(n)}

    def evidence_coverage(self, cited_ids):
        """Fraction of decision-critical constraints cited from the CURRENT corpus."""
        crit = [c["source_id"] for c in self.constraints if c["critical"]]
        cited = set(cited_ids or [])
        # a citation counts only if it names a constraint whose current spec exists
        covered = [s for s in set(crit)
                   if s in cited and self.rag.current_spec(s)["current"] is not None]
        return {"coverage": len(covered) / len(set(crit)) if crit else 1.0,
                "required": sorted(set(crit)), "cited": sorted(cited)}

    def recommendation_safe(self, rec_design, feed_z):
        """Directly SIMULATE the LLM-proposed point and require it to clear all
        constraint margins (non-negotiable #4: verify, don't trust proximity)."""
        if not rec_design:
            return {"safe": False, "min_margin": -9.99, "reason": "no recommendation"}
        r = self.eval_design(rec_design, feed_z)
        safe = bool(r["converged"] and
                    r["min_margin"] >= self.config["gate"]["rec_safe_margin"])
        return {"safe": safe, "min_margin": r["min_margin"],
                "converged": r["converged"]}

    def decision_equals_verified(self, released_x, released_obj, verified_x, verified_obj):
        g = self.config["gate"]
        if released_x is None or verified_x is None:
            return False
        for dv in self.active_vars:
            tol = g["rec_abs_tol"].get(dv["name"], 1e-6)
            if abs(float(released_x[dv["name"]]) - float(verified_x[dv["name"]])) > tol:
                return False
        if verified_obj is not None and released_obj is not None:
            if abs(released_obj - verified_obj) > g["rec_rel_tol"] * abs(verified_obj):
                return False
        return True

    def certify(self, design, feed_z, verified_x, verified_obj,
                released_x, released_obj, cited_ids, rec_design):
        g = self.config["gate"]
        r   = self.eval_design(design, feed_z)
        rob = self.robustness(design, feed_z)
        ev  = self.evidence_coverage(cited_ids)
        rs  = self.recommendation_safe(rec_design, feed_z)
        deq = self.decision_equals_verified(released_x, released_obj,
                                            verified_x, verified_obj)
        cert = AdmissibilityCertificate(
            converged=r["converged"],
            margins_ok=bool(r["converged"] and r["min_margin"] >= g["buffer"]),
            robust=bool(rob["feasible_fraction"] >= g["r_min"]),
            evidence_ok=bool(ev["coverage"] >= g["e_min"]),
            recommendation_safe=rs["safe"],
            decision_equals_verified=deq,
            margins=r["margins"],
            metrics={"min_margin": r["min_margin"],
                     "robustness": rob["feasible_fraction"],
                     "worst_robust_margin": rob["worst_min_margin"],
                     "evidence_coverage": ev["coverage"],
                     "rec_min_margin": rs["min_margin"]})
        if passes(cert):
            cert.reason = "PASS"
        else:
            fails = [n for n, v in (("converged", cert.converged),
                                    ("margins_ok", cert.margins_ok),
                                    ("robust", cert.robust),
                                    ("evidence_ok", cert.evidence_ok),
                                    ("recommendation_safe", cert.recommendation_safe),
                                    ("decision_equals_verified", cert.decision_equals_verified),
                                    ("explanation_consistent", cert.explanation_consistent))
                     if not v]
            cert.reason = "FAIL-CLOSED: " + ", ".join(fails)
        return cert

    # ── counterfactual minimal-deviation safe alternative ─────────────────────
    def counterfactual(self, u0, feed_z, rng=None):
        """
        u* = argmin || u - u0 ||_W  s.t. design is feasible AND robust.
        Solved with the configured backend (GP-BO on real Aspen, DE on the
        backend); the deviation is the objective and feasibility+robustness is
        the feasibility predicate. Returns the minimal-deviation admissible
        operating point, or no_feasible_point_found.
        """
        rng = rng if rng is not None else self.rng
        g   = self.config["gate"]
        W   = {dv["name"]: 1.0 / max(dv["upper"] - dv["lower"], 1e-9)
               for dv in self.active_vars}
        def ev(xd):
            r = self.eval_design(xd, feed_z)
            if not r["feasible"]:
                return {"objective": None, "converged": r["converged"],
                        "feasible": False, "min_margin": r["min_margin"]}
            rob = self.robustness(xd, feed_z)
            robust = rob["feasible_fraction"] >= g["r_min"]
            dev = sum((W[k] * (float(xd[k]) - float(u0[k]))) ** 2 for k in W)
            return {"objective": float(dev), "converged": r["converged"],
                    "feasible": bool(r["feasible"] and robust),
                    "min_margin": min(r["min_margin"], rob["worst_min_margin"])}
        spec = {"variables": [{"name": dv["name"], "type": dv["type"],
                               "lower": dv["lower"], "upper": dv["upper"]}
                              for dv in self.active_vars]}
        res = self.optimizer.optimize(spec, ev, rng, backend=self._backend())
        return res

    # ── verifier consensus over K candidate specifications ────────────────────
    def consensus(self, specs, feed_z, cited_ids, rec_design, rng=None,
                  backend=None):
        """Optimize + certify each candidate spec; return the best CERTIFIED
        (admissible, min-objective) design, plus all candidate records."""
        rng = rng if rng is not None else self.rng
        cands = []
        for i, spec in enumerate(specs):
            r = self.optimizer.optimize(spec, self.make_evaluate_fn(feed_z),
                                        np.random.default_rng(scenario_seed(f"cons{i}")),
                                        backend=backend or spec.get("backend"))
            if not r["feasible"]:
                cands.append({"spec_i": i, "result": r, "cert": None,
                              "admissible": False}); continue
            cert = self.certify(r["x"], feed_z, r["x"], r["objective"],
                                r["x"], r["objective"], cited_ids, rec_design)
            cands.append({"spec_i": i, "result": r, "cert": cert,
                          "admissible": passes(cert), "objective": r["objective"]})
        admissible = [c for c in cands if c["admissible"]]
        best = min(admissible, key=lambda c: c["objective"]) if admissible else None
        return {"best": best, "candidates": cands}

    # ── post-hoc ground-truth scoring (CURRENT constraints + robustness) ──────
    def _post_eval(self, design_x, feed_z):
        r   = self.eval_design(design_x, feed_z)
        rob = self.robustness(design_x, feed_z)
        robust = rob["feasible_fraction"] >= self.config["gate"]["r_min"]
        return {"nominal_feasible": bool(r["feasible"]), "robust": bool(robust),
                "feasible_post": bool(r["feasible"] and robust),
                "robustness": rob["feasible_fraction"],
                "duty": r["outputs"].get("reboiler_duty")}

    def _ts_from_history(self, history):
        """Per-iteration time series for figures: running best-feasible objective,
        min margin, feasibility, cumulative evaluation count (logged data only)."""
        ts, best = [], None
        for i, h in enumerate(history):
            if h["feasible"] and h["objective"] is not None:
                best = h["objective"] if best is None else min(best, h["objective"])
            ts.append({"eval": i + 1, "objective": h["objective"],
                       "best_feasible_objective": best, "min_margin": h["min_margin"],
                       "feasible": h["feasible"], "converged": h["converged"]})
        return ts

    def _result(self, case, system, **kw):
        tr = self.flowsheet.trace                      # this run's audit trace
        trace_cap = (tr[:150] + tr[-50:]) if len(tr) > 200 else list(tr)
        base = {"case_id": case["id"], "system": system, "feed_z": case["feed_z"],
                "aspen_trace": trace_cap, "aspen_trace_total": len(tr),
                "released": False, "abstained": False, "counterfactual_used": False,
                "decision": None, "objective": None, "certificate": None,
                "cert_passes": False, "cited": [], "evidence_coverage": None,
                "feasible_post": None, "nominal_feasible": None, "robust": None,
                "violates_current": None, "energy_improvement": None,
                "n_aspen_eval": len(self.flowsheet.trace), "wall_s": None,
                "time_series": [], "reason": "", "rationale": "", "model": "mock",
                "capture": None}
        # Carry the repeat identity (D4 decision-stability): with_repeats() gives each repeat a
        # distinct case_id (..._r0/_r1) and stamps base_case_id/rep/seed. Propagate them so
        # decision_stability groups repeats of the SAME base case together (grouping on the
        # per-rep case_id would isolate every rep -> n_cases=0). Absent for single-rep runs, so
        # the record is byte-identical to the locked campaign when FCO_CAMPAIGN_REPS=1.
        for _k in ("base_case_id", "rep", "seed"):
            if _k in case:
                base[_k] = case[_k]
        base.update(kw)
        return base

    def baseline_duty(self, feed_z):
        """Reference duty of the starting baseline design (RR 2.5, D:F 0.5, fs 13)."""
        out = self.flowsheet.evaluate({"reflux_ratio": 2.5, "DtoF": 0.5,
                                       "feed_stage": 13}, feed_z)
        return out.get("reboiler_duty")

    def _runner_capture(self, plan, *, consensus=None, chosen=None,
                        released_point=None, post=None, feed_z=None, explainer=None):
        """Assemble the per-(case,arm) DERIVED audit for the one-shot capture (§2): the
        plan-side capture (prompt/retrieved/decision/hallucinated/meta/ground_truth, from
        plan()) + the K-candidate consensus scores + the chosen design's certificate +
        the verified Aspen outputs at decision time. Stored on the result record
        (harness_progress.jsonl); joins the verbatim inputs in llm_raw_log.jsonl on
        (case_id, system). All deterministic re-derivations -> $0-regenerable, persisted
        so the one-shot paid run never depends on replay."""
        cap = dict((plan or {}).get("capture") or {})
        # the verbatim prompt + retrieved set live in llm_raw_log.jsonl (joinable on
        # case_id+system); keep the result record lean by not duplicating them here.
        cap.pop("prompt", None)
        cap.pop("retrieved", None)
        if consensus is not None:
            cap["consensus"] = consensus
        vo = None
        if chosen is not None and feed_z is not None:
            x = chosen["result"]["x"]
            e = self.eval_design(x, feed_z)
            vo = {"design": x, "outputs": e["outputs"], "feasible": bool(e["feasible"]),
                  "min_margin": e.get("min_margin"),
                  "certificate": (asdict(chosen["cert"]) if chosen.get("cert") else None)}
        elif released_point is not None and feed_z is not None:
            e = self.eval_design(released_point, feed_z)
            vo = {"design": released_point, "outputs": e["outputs"],
                  "feasible": bool(e["feasible"]), "min_margin": e.get("min_margin")}
        if vo is not None:
            cap["verified_outputs"] = vo
        if post is not None:
            cap["post_eval"] = post
        if explainer is not None:
            cap["explainer"] = {
                "narrative": explainer.get("narrative"),
                "stated_action": explainer.get("stated_action"),
                "cited_constraints": explainer.get("cited_constraints"),
                "model": explainer.get("model"),
                "explanation_consistent": (chosen or {}).get("cert").explanation_consistent
                if (chosen and chosen.get("cert")) else None}
        return cap

    # ── FULL SYSTEM (and ablations) ───────────────────────────────────────────
    def run_case(self, case, mode="full"):
        """
        LLM-planned, RAG-grounded, optimizer-solved, gate-certified pipeline.
        Releases ONLY a certified design (== verified optimizer result); on a
        gate failure it tries the counterfactual safe alternative, else abstains.
        `mode` selects ablations: full | ablate_rag | ablate_gate | ablate_robust
        | ablate_counterfactual | ablate_consensus1 | no_rag (the MEASURED RAG
        ablation: strip the retrieved context from the prompt, keep the gate -> P2).
        """
        t0 = time.time()
        feed_z = case["feed_z"]
        self.flowsheet.trace = []
        use_gate   = mode != "ablate_gate"
        use_robust = mode != "ablate_robust"
        use_cf     = mode != "ablate_counterfactual"
        use_rag    = mode != "ablate_rag"
        rag_on     = mode != "no_rag"                  # measured RAG ablation (P2)
        backend    = self._backend()                  # gp_bo on real Aspen, DE on mock

        plan = self.plan(case, system=mode, rag_on=rag_on)
        if not plan["ok"]:
            return self._result(case, mode, abstained=True,
                                reason=("ungrounded_spec" if plan.get("ungrounded")
                                        else "planning_failed"),
                                wall_s=round(time.time()-t0, 3), model=plan.get("model", "mock"),
                                capture=self._runner_capture(plan))
        decision = plan["decision"]
        specs    = plan["specs"]
        if mode == "ablate_consensus1":
            specs = specs[:1]
        cited = (decision.get("cited_constraints", []) if use_rag else [])
        rec   = decision.get("recommendation")
        bduty = self.baseline_duty(feed_z)

        # consensus: optimize each candidate spec (under ITS OWN constraints —
        # stale in the stale-spec case), then certify (gate uses CURRENT corpus).
        cands = []
        for j, s in enumerate(specs):
            prob = self._spec_to_problem(s, backend)
            rng  = np.random.default_rng(scenario_seed(f"{case['id']}|{mode}|{j}"))
            r = self.optimizer.optimize(prob, self.make_evaluate_fn(feed_z, s["constraints"]),
                                        rng, backend=backend)
            cand = {"spec": s, "result": r, "cert": None, "admissible": False}
            if r["feasible"]:
                cert = self.certify(r["x"], feed_z, r["x"], r["objective"],
                                    r["x"], r["objective"], cited, rec)
                if use_robust is False:
                    cert.robust = True                    # ablation: drop robustness req
                cand["cert"] = cert
                cand["admissible"] = (True if not use_gate else passes(cert))
            cands.append(cand)

        admissible = [c for c in cands if c["admissible"] and c["result"]["feasible"]]
        chosen = (min(admissible, key=lambda c: c["result"]["objective"])
                  if admissible else None)

        # fail-closed: try the counterfactual safe alternative from the best
        # feasible-but-inadmissible optimizer result.
        cf_used = False
        if chosen is None and use_gate and use_cf:
            seed_pts = [c for c in cands if c["result"]["feasible"]]
            if seed_pts:
                u0  = min(seed_pts, key=lambda c: c["result"]["objective"])["result"]["x"]
                cf  = self.counterfactual(u0, feed_z,
                                          rng=np.random.default_rng(scenario_seed(case["id"]+"cf")))
                if cf["feasible"]:
                    cf_obj = self.eval_design(cf["x"], feed_z)["outputs"]["reboiler_duty"]
                    cert = self.certify(cf["x"], feed_z, cf["x"], cf_obj,
                                        cf["x"], cf_obj, cited, rec)
                    if passes(cert):
                        cf_used = True
                        chosen = {"spec": None, "result": {
                            "x": cf["x"], "objective": cf_obj,
                            "history": cf.get("history", [])}, "cert": cert,
                            "admissible": True}

        consensus = [{"spec_i": j, "feasible": bool(c["result"].get("feasible")),
                      "objective": c["result"].get("objective"),
                      "admissible": bool(c.get("admissible")),
                      "certificate": (asdict(c["cert"]) if c.get("cert") else None)}
                     for j, c in enumerate(cands)]
        consensus_done = consensus
        ev = self.evidence_coverage(cited)
        if chosen is None:
            binding = (cands[0]["cert"].reason if cands and cands[0]["cert"]
                       else "no feasible/admissible design")
            return self._result(case, mode, abstained=True, reason=f"fail_closed: {binding}",
                                cited=cited, evidence_coverage=ev["coverage"],
                                wall_s=round(time.time()-t0, 3), model=plan["model"],
                                capture=self._runner_capture(plan, consensus=consensus_done))
        x, obj = chosen["result"]["x"], chosen["result"]["objective"]
        # EXPLAINER (call 2): operator narrative for the RELEASED design; its faithfulness
        # is folded INTO the certificate (post-hoc, no Aspen re-run). An unfaithful
        # narrative (wrong action, or a fabricated/un-retrieved citation) is a
        # narrative-mismatch hallucination -> fail-closed. Mock explainer is faithful, so
        # the gated decision is unchanged (result-neutral). Skipped when the gate is off.
        explainer, narrative = None, ""
        if use_gate:
            explainer = self.explain(case, x, decision, plan["rationale_passages"],
                                     plan["spec_constraints"], system=mode)
            rids = self._explainer_retrieved_ids(plan["rationale_passages"],
                                                 plan["spec_constraints"])
            chosen["cert"] = self._augment_cert_with_explainer(chosen["cert"], explainer, x, rids)
            narrative = explainer.get("narrative") or ""
            if not passes(chosen["cert"]):
                return self._result(case, mode, abstained=True,
                                    reason=f"fail_closed: {chosen['cert'].reason}",
                                    cited=cited, evidence_coverage=ev["coverage"],
                                    rationale=narrative, wall_s=round(time.time()-t0, 3),
                                    model=plan["model"],
                                    capture=self._runner_capture(plan, consensus=consensus_done,
                                                                 explainer=explainer))
        post = self._post_eval(x, feed_z)
        return self._result(
            case, mode, released=True, counterfactual_used=cf_used,
            decision=x, objective=obj, certificate=asdict(chosen["cert"]) if chosen["cert"] else None,
            cert_passes=(passes(chosen["cert"]) if chosen["cert"] else (not use_gate)),
            cited=cited, evidence_coverage=ev["coverage"],
            feasible_post=post["feasible_post"], nominal_feasible=post["nominal_feasible"],
            robust=post["robust"], violates_current=(not post["nominal_feasible"]),
            energy_improvement=((bduty - obj) / bduty if (bduty and obj) else None),
            wall_s=round(time.time()-t0, 3), rationale=narrative,
            time_series=self._ts_from_history(chosen["result"].get("history", [])),
            model=plan["model"],
            capture=self._runner_capture(plan, consensus=consensus_done, chosen=chosen,
                                         post=post, feed_z=feed_z, explainer=explainer))

    # ── BASELINES (symmetric information; no oracle) ──────────────────────────
    def run_b0(self, case):
        """B0 — ungated LLM: release the LLM's proposed operating point directly;
        verify post hoc for SCORING ONLY (no Aspen-verified gate)."""
        t0 = time.time(); self.flowsheet.trace = []
        plan = self.plan(case, system="B0_ungated_llm")
        rec  = plan["decision"]["recommendation"] if plan["ok"] else None
        if rec is None:
            return self._result(case, "B0_ungated_llm", abstained=True,
                                reason="no recommendation", wall_s=round(time.time()-t0, 3),
                                model=plan.get("model", "mock"),
                                capture=self._runner_capture(plan))
        post  = self._post_eval(rec, case["feed_z"])
        bduty = self.baseline_duty(case["feed_z"])
        return self._result(case, "B0_ungated_llm", released=True, decision=rec,
                            objective=post["duty"], feasible_post=post["feasible_post"],
                            nominal_feasible=post["nominal_feasible"], robust=post["robust"],
                            violates_current=(not post["nominal_feasible"]),
                            energy_improvement=((bduty-post["duty"])/bduty if (bduty and post["duty"]) else None),
                            cited=plan["decision"].get("cited_constraints", []) if plan["ok"] else [],
                            wall_s=round(time.time()-t0, 3), reason="released ungated (post-hoc scored)",
                            model=plan["model"] if plan["ok"] else "mock",
                            capture=self._runner_capture(plan, released_point=rec,
                                                         post=post, feed_z=case["feed_z"]))

    def run_b1(self, case):
        """B1 — LLM+RAG+optimizer, NO gate: release the optimizer result (under
        the spec's constraints) without certification."""
        t0 = time.time(); self.flowsheet.trace = []
        plan = self.plan(case, system="B1_llm_rag_nogate")
        if not plan["ok"]:
            return self._result(case, "B1_llm_rag_nogate", abstained=True,
                                reason=("ungrounded_spec" if plan.get("ungrounded")
                                        else "planning_failed"),
                                wall_s=round(time.time()-t0, 3),
                                model=plan.get("model", "mock"),
                                capture=self._runner_capture(plan))
        s = plan["specs"][0]
        bk = self._backend()
        r = self.optimizer.optimize(self._spec_to_problem(s, bk),
                                    self.make_evaluate_fn(case["feed_z"], s["constraints"]),
                                    np.random.default_rng(scenario_seed(case["id"]+"b1")),
                                    backend=bk)
        if not r["feasible"]:
            return self._result(case, "B1_llm_rag_nogate", abstained=True,
                                reason="optimizer found no feasible point",
                                wall_s=round(time.time()-t0, 3), model=plan["model"],
                                capture=self._runner_capture(plan))
        post  = self._post_eval(r["x"], case["feed_z"]); bduty = self.baseline_duty(case["feed_z"])
        return self._result(case, "B1_llm_rag_nogate", released=True, decision=r["x"],
                            objective=r["objective"], feasible_post=post["feasible_post"],
                            nominal_feasible=post["nominal_feasible"], robust=post["robust"],
                            violates_current=(not post["nominal_feasible"]),
                            energy_improvement=((bduty-r["objective"])/bduty if (bduty and r["objective"]) else None),
                            cited=plan["decision"].get("cited_constraints", []),
                            wall_s=round(time.time()-t0, 3),
                            time_series=self._ts_from_history(r.get("history", [])),
                            reason="released ungated optimizer result", model=plan["model"],
                            capture=self._runner_capture(plan, released_point=r["x"],
                                                         post=post, feed_z=case["feed_z"]))

    def run_b2(self, case):
        """B2 — optimizer + gate, NO LLM/RAG (fixed expert bounds + CURRENT
        constraints). The control that isolates the LLM/RAG contribution."""
        t0 = time.time(); self.flowsheet.trace = []
        feed_z = case["feed_z"]; bduty = self.baseline_duty(feed_z)
        crit = [c["source_id"] for c in self.constraints if c["critical"]]   # expert knows the specs
        bk = self._backend()
        prob = {"variables": [{"name": dv["name"], "type": dv["type"],
                               "lower": dv["lower"], "upper": dv["upper"]}
                              for dv in self.active_vars], "backend": bk}
        r = self.optimizer.optimize(prob, self.make_evaluate_fn(feed_z),  # CURRENT constraints
                                    np.random.default_rng(scenario_seed(case["id"]+"b2")),
                                    backend=bk)
        chosen_x, chosen_obj, cf_used = None, None, False
        if r["feasible"]:
            cert = self.certify(r["x"], feed_z, r["x"], r["objective"],
                                r["x"], r["objective"], crit, r["x"])  # rec = own result
            if passes(cert):
                chosen_x, chosen_obj = r["x"], r["objective"]
        if chosen_x is None and r["feasible"]:
            cf = self.counterfactual(r["x"], feed_z,
                                     rng=np.random.default_rng(scenario_seed(case["id"]+"b2cf")))
            if cf["feasible"]:
                xcf = cf["x"]; ocf = self.eval_design(xcf, feed_z)["outputs"]["reboiler_duty"]
                cert = self.certify(xcf, feed_z, xcf, ocf, xcf, ocf, crit, xcf)
                if passes(cert):
                    chosen_x, chosen_obj, cf_used = xcf, ocf, True
        if chosen_x is None:
            return self._result(case, "B2_optimizer_gate", abstained=True,
                                reason="fail_closed (no admissible/robust design)",
                                cited=crit, wall_s=round(time.time()-t0, 3), model="none",
                                capture={"system": "B2_optimizer_gate", "model": "none",
                                         "rag_on": False, "note": "no LLM call (control arm)"})
        post = self._post_eval(chosen_x, feed_z)
        return self._result(case, "B2_optimizer_gate", released=True, counterfactual_used=cf_used,
                            decision=chosen_x, objective=chosen_obj,
                            feasible_post=post["feasible_post"], nominal_feasible=post["nominal_feasible"],
                            robust=post["robust"], violates_current=(not post["nominal_feasible"]),
                            energy_improvement=((bduty-chosen_obj)/bduty if (bduty and chosen_obj) else None),
                            cited=crit, evidence_coverage=1.0,
                            wall_s=round(time.time()-t0, 3),
                            time_series=self._ts_from_history(r.get("history", [])), model="none",
                            capture={"system": "B2_optimizer_gate", "model": "none",
                                     "rag_on": False, "note": "no LLM call (control arm)",
                                     "verified_outputs": {"design": chosen_x,
                                                          "outputs": self.eval_design(chosen_x, feed_z)["outputs"]},
                                     "post_eval": post})


def build_cases(config=CONFIG):
    """Pre-registered scenario ensemble: nominal, feed-drift, stale-spec."""
    z0 = config["system"]["feed"]["z_methanol"]
    cases = [{"id": "C0_nominal", "name": "Nominal methanol-water split",
              "feed_z": z0, "stale_spec": False}]
    for z in (0.45, 0.55):     # feed-drift scenarios (optimizer re-optimizes D:F)
        cases.append({"id": f"C_drift_{int(z*100)}", "name": f"Feed drift z={z}",
                      "feed_z": z, "stale_spec": False})
    cases.append({"id": "C_stale_spec", "name": "Stale purity-spec injection",
                  "feed_z": z0, "stale_spec": True})
    if os.environ.get("FCO_ADVERSARIAL"):
        # Re-run only (OFF by default -> locked set unchanged): OUT-OF-CORPUS trap. The
        # distillate-purity spec is dropped from the corpus, so the model cannot ground it
        # and must abstain or fabricate; the gate still owns the true limit and fails closed.
        cases.append({"id": "C_ooc_dist_purity", "name": "Out-of-corpus distillate-purity spec",
                      "feed_z": z0, "stale_spec": False, "drop_sources": ["SPEC-DIST-PURITY"]})
    return cases


def build_campaign_cases(config=CONFIG):
    """COMPREHENSIVE campaign scenarios (P4): a feed-drift GRID + a DISTRIBUTION of stale
    traps + slack precision checks (not a single anecdote). Used by campaign.py, NOT by
    the locked study (build_cases stays the pre-registered 4). The wide feed grid needs
    the widened D:F bounds the campaign config sets (a cut far from 0.5 needs D:F ~= z)."""
    z0 = config["system"]["feed"]["z_methanol"]
    cases = [{"id": "C0_nominal", "name": "Nominal methanol-water split",
              "feed_z": z0, "stale_spec": False}]
    # FINER feed-drift grid (z 0.35-0.65, ~0.02-0.03 spacing) -> ~12 independent drift items
    # instead of 6, for statistical power (D1). int(z*100) ids stay unique at this spacing.
    for z in (0.35, 0.38, 0.40, 0.42, 0.45, 0.47, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65):
        cases.append({"id": f"C_drift_{int(round(z*100))}", "name": f"Feed drift z={z}",
                      "feed_z": z, "stale_spec": False})
    # stale-trap DISTRIBUTION on the BINDING purity specs (each BITES -> caught by the gate)
    cases.append({"id": "C_stale_spec", "name": "Stale purity-spec (both)",
                  "feed_z": z0, "stale_spec": True})
    cases.append({"id": "C_stale_dist", "name": "Stale distillate-purity (partial)",
                  "feed_z": z0, "stale_sources": ["SPEC-DIST-PURITY"]})
    cases.append({"id": "C_stale_bot", "name": "Stale bottoms-purity (partial)",
                  "feed_z": z0, "stale_sources": ["SPEC-BOT-PURITY"]})
    # stale traps at OFF-nominal feed (the trap and the drift bind together -> more biting items)
    cases.append({"id": "C_stale_spec_d40", "name": "Stale purity-spec at z=0.40",
                  "feed_z": 0.40, "stale_spec": True})
    cases.append({"id": "C_stale_spec_d60", "name": "Stale purity-spec at z=0.60",
                  "feed_z": 0.60, "stale_spec": True})
    # stale on SLACK constraints (flooding/duty): the min-duty optimum sits far inside, so
    # the gate correctly RELEASES (a precision check that the gate is not trigger-happy).
    cases.append({"id": "C_slack_flood", "name": "Stale flooding limit (slack)",
                  "feed_z": z0, "stale_sources": ["LIM-FLOODING"]})
    cases.append({"id": "C_slack_duty", "name": "Stale duty limit (slack)",
                  "feed_z": z0, "stale_sources": ["LIM-REB-DUTY"]})
    cases.append({"id": "C_slack_dt", "name": "Stale min-approach limit (slack)",
                  "feed_z": z0, "stale_sources": ["LIM-DT-MIN"]})
    return cases


def with_repeats(cases, reps=None):
    """Expand each case into `reps` repeats with unique ids + per-repeat seeds, for the
    decision-stability metric (D4) and clustered statistical power (D1). reps defaults to the
    FCO_CAMPAIGN_REPS env (1 = no expansion, byte-identical to the locked single-rep campaign).
    The repeat tag/seed lets the live API's residual temperature-0 nondeterminism surface as
    proposal variation while the gate-released design stays fixed (the CS2 analog of CS1 fig3)."""
    if reps is None:
        reps = int(os.environ.get("FCO_CAMPAIGN_REPS", "1"))
    if reps <= 1:
        return list(cases)
    out = []
    for c in cases:
        for r in range(reps):
            cc = dict(c)
            cc["rep"] = r
            cc["seed"] = 1000 * r + (abs(hash(c["id"])) % 1000)
            cc["id"] = "%s__r%d" % (c["id"], r)
            cc["base_case_id"] = c["id"]
            out.append(cc)
    return out


ABLATIONS = ["ablate_rag", "ablate_gate", "ablate_robust",
             "ablate_counterfactual", "ablate_consensus1"]


def run_harness(agent, cases, save_dir=RESULT_DIR, resume=True):
    """Run B0/B1/B2 + full + ablations over the pre-registered scenarios.

    INCREMENTAL PERSISTENCE (R3): every (case, system) result is appended to
    harness_progress.jsonl and flushed immediately, so a crash mid-sweep keeps all
    prior work. On resume, completed (case_id, system) pairs are reused instead of
    re-run — essential for the slow real-Aspen sweep. Start a CLEAN run by deleting
    harness_progress.jsonl first (resume reuses whatever is already there)."""
    progress = os.path.join(save_dir, "harness_progress.jsonl")
    done = {}
    if resume and os.path.exists(progress):
        with open(progress) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rr = json.loads(line)
                    done[(rr["case_id"], rr["system"])] = rr
                except Exception as err:                  # never let a bad line abort
                    diag_log.warning("harness_progress parse skip: %s", err)
        if done:
            diag_log.info("harness resume: %d completed (case,system) results reused",
                          len(done))
    results = []
    fh = open(progress, "a")
    aborted = None
    try:
        def step(case, system, thunk):
            if (case["id"], system) in done:
                results.append(done[(case["id"], system)])
                return
            r = thunk()
            results.append(r)
            fh.write(json.dumps(r, default=str) + "\n"); fh.flush()
        for case in cases:
            steps = [("B0_ungated_llm",   lambda c=case: agent.run_b0(c)),
                     ("B1_llm_rag_nogate", lambda c=case: agent.run_b1(c)),
                     ("B2_optimizer_gate", lambda c=case: agent.run_b2(c)),
                     ("full",             lambda c=case: agent.run_case(c, mode="full"))]
            steps += [(ab, (lambda c=case, ab=ab: agent.run_case(c, mode=ab)))
                      for ab in ABLATIONS]
            for system, thunk in steps:
                try:
                    step(case, system, thunk)
                except Exception as err:
                    # A live API failure / truncation STOPS the run cleanly (all prior
                    # results are flushed -> resumable). In mock mode this is a real bug,
                    # so re-raise. Live errors are already captured to llm_raw_log.
                    if agent.mock_llm:
                        raise
                    aborted = {"reason": ("truncation"
                                          if isinstance(err, LLMTruncatedError) else "api_error"),
                               "error": f"{type(err).__name__}: {err}",
                               "at": [case["id"], system]}
                    break
                # LIVE contamination abort: stop paying if the LLM produces garbage.
                if not agent.mock_llm:
                    cr = agent.contamination_report()
                    if (cr["total_live_plans"] >= LLM_CONTAMINATION_MIN_N and
                            cr["rate"] >= LLM_CONTAMINATION_ABORT):
                        aborted = {"reason": "contamination", **cr,
                                   "at": [case["id"], system]}
                        break
            if aborted:
                break
    finally:
        fh.close()
    if aborted:
        diag_log.error("run_harness ABORT (%s) at %s: %s",
                       aborted["reason"], aborted.get("at"), aborted)
        print("  [ABORT:%s] stopping the run after %d results (resumable): %s"
              % (aborted["reason"], len(results), aborted.get("error", aborted)))
    return results


def compute_metrics(results):
    """Aggregate the paper metrics per system (driven entirely by logged results)."""
    by_sys = defaultdict(list)
    for r in results:
        by_sys[r["system"]].append(r)
    metrics = {}
    for sys_name, rs in by_sys.items():
        rel   = [r for r in rs if r["released"]]
        # the stale-TRAP bucket = the biting purity-stale cases (C_stale_*); slack
        # precision checks are named C_slack_* and excluded so the headline stays clean.
        stale = [r for r in rs if str(r["case_id"]).startswith("C_stale")]
        stale_rel = [r for r in stale if r["released"]]
        feas_rel  = [r for r in rel if r["feasible_post"]]
        drift_rel = [r for r in rel if str(r["case_id"]).startswith("C_drift")]
        improvements = [r["energy_improvement"] for r in feas_rel
                        if r["energy_improvement"] is not None]
        covered = [r for r in rel if (r["evidence_coverage"] or 0) >= 1.0]
        metrics[sys_name] = {
            "n_cases": len(rs), "n_released": len(rel), "n_abstained": len(rs) - len(rel),
            # feasible_solution_rate = feasible_post = (nominal-feasible AND robust);
            # reported alongside its two components so the gate's value is precise
            # (e.g. ablate_gate is ~nominally feasible but NOT robust, not "infeasible").
            "feasible_solution_rate": (len(feas_rel) / len(rel)) if rel else None,
            "nominal_feasible_rate": (sum(1 for r in rel if r.get("nominal_feasible"))
                                      / len(rel)) if rel else None,
            "robust_release_rate": (sum(1 for r in rel if r.get("robust")) / len(rel))
                                   if rel else None,
            # HEADLINE: feasibility (nominal AND robust) specifically UNDER FEED DRIFT
            "feasible_rate_under_drift": (sum(1 for r in drift_rel if r["feasible_post"])
                                          / len(drift_rel)) if drift_rel else None,
            # HEADLINE: false-release under stale knowledge
            "false_release_rate_stale": (sum(1 for r in stale_rel if r["violates_current"])
                                         / len(stale_rel)) if stale_rel else 0.0,
            # energy vs the NOMINAL starting design (one of two baselines; see
            # cross_system() for the vs-B0 "cost of robustness" framing)
            "mean_energy_improvement_vs_nominal": (round(float(np.mean(improvements)), 4)
                                                   if improvements else None),
            "mean_energy_improvement": (round(float(np.mean(improvements)), 4)
                                        if improvements else None),   # back-compat alias
            "all_released_cited_rate": (len(covered) / len(rel)) if rel else None,
            "total_aspen_evals": int(sum(r["n_aspen_eval"] for r in rs)),
            "mean_wall_s": round(float(np.mean([r["wall_s"] for r in rs if r["wall_s"]])), 3),
        }
    return metrics


def cross_system(results):
    """R5 cross-system comparisons from PERSISTED results (no Aspen re-run):
    (a) TWO-BASELINE energy framing — full vs the nominal starting design AND vs
    the ungated-LLM (B0) released point ("cost of robustness"); never a single
    ambiguous number. (b) full vs B2 (BOTH gated) on robustness + false-release,
    isolating the LLM/RAG (retrieval+planning) contribution over plain optimization."""
    by = defaultdict(dict)
    for r in results:
        by[r["case_id"]][r["system"]] = r
    energy, fvb2 = [], []
    for cid, d in sorted(by.items()):
        full = d.get("full", {}); b0 = d.get("B0_ungated_llm", {}); b2 = d.get("B2_optimizer_gate", {})
        fobj = full.get("objective") if full.get("released") else None
        dEn  = full.get("energy_improvement")
        # recover the nominal starting-design duty: dE = (base - obj)/base
        base = (fobj / (1.0 - dEn)) if (fobj and dEn is not None and dEn not in (1.0,)) else None
        b0obj = b0.get("objective") if b0.get("released") else None
        # SIGN CONVENTION (guardrails 3.7): dE_vs_nominal=(base-duty)/base and
        # dE_vs_b0=(B0_duty-full_duty)/B0_duty -> POSITIVE means full is cheaper/better
        # (lower duty). Counterintuitive for the B0 comparison; stated by the field/figure.
        energy.append({
            "case_id": cid,
            "full_duty_kW":       round(fobj, 1) if fobj else None,
            "nominal_baseline_kW": round(base, 1) if base else None,
            "dE_vs_nominal":      round(dEn, 4) if dEn is not None else None,
            "b0_released_duty_kW": round(b0obj, 1) if b0obj else None,
            "dE_vs_b0":           (round((b0obj - fobj) / b0obj, 4) if (fobj and b0obj) else None),
        })
        fvb2.append({"case_id": cid,
            "full": {"released": full.get("released", False), "feasible": full.get("feasible_post"),
                     "robust": full.get("robust"), "violates_current": full.get("violates_current")},
            "b2":   {"released": b2.get("released", False), "feasible": b2.get("feasible_post"),
                     "robust": b2.get("robust"), "violates_current": b2.get("violates_current")}})
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 4) if xs else None
    return {
        "energy_two_baseline": energy,
        "mean_dE_vs_nominal": _mean([e["dE_vs_nominal"] for e in energy]),
        "mean_dE_vs_b0": _mean([e["dE_vs_b0"] for e in energy]),
        "full_vs_b2": fvb2,
    }


def rag_value(results, with_arm="full", without_arm="no_rag"):
    """MEASURED RAG contribution (P2 / claim C3): compare the WITH-RAG arm to the
    WITHOUT-RAG arm (same gate; the retrieved context is stripped from the prompt) on the
    hallucination the gate must catch + grounding + safety. Driven ENTIRELY by the
    captured per-call records (no re-run, no assertion):
      - invented_threshold_rate: fraction of plans with >=1 hallucinated (invented)
        constraint threshold (the citation gate's count);
      - grounding_coverage: mean evidence coverage on released designs;
      - false_release_rate_stale: fraction of stale-case RELEASES that violate the current
        spec (an unsafe release).
    RAG 'helps' iff invented_threshold_rate (and any false-release) FALL with RAG vs
    without -- so the delta isolates RAG's contribution rather than asserting it."""
    def _agg(arm):
        rs = [r for r in results if r["system"] == arm]
        if not rs:
            return None
        halluc = [(r.get("capture") or {}).get("hallucinated") for r in rs]
        halluc = [h for h in halluc if h is not None]
        rel = [r for r in rs if r["released"]]
        cov = [r["evidence_coverage"] for r in rel if r.get("evidence_coverage") is not None]
        stale_rel = [r for r in rs if str(r["case_id"]).startswith("C_stale") and r["released"]]
        return {
            "n_plans": len(rs), "n_released": len(rel), "n_abstained": len(rs) - len(rel),
            "invented_threshold_rate": (round(sum(1 for h in halluc if h > 0) / len(halluc), 3)
                                        if halluc else None),
            "mean_hallucinated_citations": (round(float(np.mean(halluc)), 3) if halluc else None),
            "grounding_coverage": (round(float(np.mean(cov)), 3) if cov else None),
            "false_release_rate_stale": (round(sum(1 for r in stale_rel
                                                   if r.get("violates_current")) / len(stale_rel), 3)
                                         if stale_rel else 0.0),
        }
    w, wo = _agg(with_arm), _agg(without_arm)
    out = {"with_arm": with_arm, "without_arm": without_arm, "with_rag": w, "without_rag": wo}
    if w and wo:
        out["delta_without_minus_with"] = {
            k: (round(wo[k] - w[k], 3) if (w.get(k) is not None and wo.get(k) is not None)
                else None)
            for k in ("invented_threshold_rate", "grounding_coverage",
                      "false_release_rate_stale")}
    return out


def model_decoupling(results, gated_arm="full", ungated_arm="B0_ungated_llm"):
    """MEASURED model-independence of the guarantee (P3 / claim C4): across planner
    models, the GATED arm's released-unsafe rate stays 0 even as a weaker model
    hallucinates / follows the stale spec more often. Driven by the captured per-call
    records (model + hallucinated + violates_current). One row per (arm, model)."""
    rows = []
    for arm in (gated_arm, ungated_arm):
        by_model = defaultdict(list)
        for r in results:
            # the campaign labels a weaker-model arm "<arm>@weak"; group both under <arm>.
            if str(r["system"]).split("@")[0] == arm:
                by_model[r.get("model", "?")].append(r)
        for model, rs in sorted(by_model.items()):
            halluc = [(r.get("capture") or {}).get("hallucinated") for r in rs]
            halluc = [h for h in halluc if h is not None]
            rel = [r for r in rs if r["released"]]
            unsafe = [r for r in rel if r.get("violates_current")]
            rows.append({
                "arm": arm, "model": model, "n_plans": len(rs), "n_released": len(rel),
                "invented_threshold_rate": (round(sum(1 for h in halluc if h > 0) / len(halluc), 3)
                                            if halluc else None),
                "released_unsafe_rate": (round(len(unsafe) / len(rel), 3) if rel else 0.0)})
    gated_unsafe = [row["released_unsafe_rate"] for row in rows if row["arm"] == gated_arm]
    return {"rows": rows, "gated_arm": gated_arm, "ungated_arm": ungated_arm,
            "gated_released_unsafe_all_models": (max(gated_unsafe) if gated_unsafe else 0.0),
            "headline": "the gated released-unsafe rate is 0 across every planner model"}


def _file_sha256(path):
    """SHA-256 of any file, or None if absent. Used both for the DETERMINISTIC
    baseline anchor (the canonical .inp) and the .bkp PROVENANCE hash."""
    if os.path.exists(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    return None


# Backward-compatible alias (build_baseline.py historically imported _bkp_sha256).
_bkp_sha256 = _file_sha256

# Canonical, DETERMINISTIC baseline anchor. The authored Aspen input language at
# case2_flowsheet/aspen/baseline_meoh_water.inp is byte-stable across rebuilds and
# tracked in git, so the baseline reconstructs from it via build_baseline.py WITHOUT
# the .bkp. Reproducibility is verified against this .inp hash + the converged
# outputs (duty/xD/xB) — NOT the .bkp hash (a .bkp embeds a save-timestamp/host
# metadata, so its hash changes on every rebuild; verified by two identical builds).
BASELINE_INP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "aspen", "baseline_meoh_water.inp")


def write_manifest(agent, cases, metrics, save_dir):
    """Run manifest (Phase 7): seeds, library versions, resolved model id, Aspen
    version + property method + components, baseline .bkp hash, ALL thresholds,
    and the active backends — the reproducibility/audit record the paper promises."""
    import platform
    versions = {}
    for mod, attr in (("numpy", "__version__"), ("scipy", "__version__"),
                      ("sklearn", "__version__"), ("matplotlib", "__version__"),
                      ("anthropic", "__version__")):
        try:
            versions[mod] = getattr(__import__(mod), attr)
        except Exception:
            versions[mod] = "n/a"
    try:
        import win32com; versions["pywin32"] = getattr(win32com, "__version__", "installed")
    except Exception:
        versions["pywin32"] = "n/a"
    bkp = agent.config["system"]["baseline_bkp"]
    manifest = {
        "base_seed": BASE_SEED, "cases": [c["id"] for c in cases],
        "mock_aspen": MOCK_ASPEN, "mock_llm": MOCK_LLM,
        "llm_model": ("mock" if MOCK_LLM else agent.model),
        "aspen": {"backend": agent.flowsheet.backend,
                  "version": ("unavailable" if MOCK_ASPEN
                              else "Aspen Plus V14 (Apwn.Document.40.0)"),
                  "property_method": agent.config["system"]["property_method"],
                  "components": agent.config["system"]["components"],
                  "hydraulics": agent.config["system"].get("hydraulics", {})},
        # CANONICAL anchor (deterministic, tracked in git): verify reproducibility
        # against this .inp hash + the converged outputs (duty/xD/xB), per HOW_TO_RUN.
        "baseline_inp": BASELINE_INP,
        "baseline_inp_sha256": (_file_sha256(BASELINE_INP) or "absent"),
        "baseline_bkp": bkp,
        # PROVENANCE ONLY (non-reproducible): a .bkp is a binary archive embedding a
        # save-timestamp/host metadata, so its hash differs on every rebuild even when
        # the model is byte-identical. Recorded for provenance; NEVER a repro check.
        "baseline_bkp_sha256_provenance": (_file_sha256(bkp) or
                                           "absent (rebuild via build_baseline.py)"),
        "rag_dense_backend": agent.rag.dense_backend,
        # Pipeline provenance: tool-use on/off + dense on/off, so the run is self-describing.
        "config": {"tool_use": bool(os.environ.get("FCO_TOOL_USE")),
                   "mcp_only": bool(os.environ.get("FCO_MCP_ONLY")),
                   "rag_dense": bool(os.environ.get("RAG_DENSE")),
                   "adversarial": bool(os.environ.get("FCO_ADVERSARIAL")),
                   "arms_filter": os.environ.get("FCO_ARMS") or None,
                   "scenarios_filter": os.environ.get("FCO_SCENARIOS") or None,
                   "tooluse_max_tokens": __import__("case2_flowsheet.mcp_server",
                                                    fromlist=["TOOLUSE_MAX_TOKENS"]).TOOLUSE_MAX_TOKENS},
        "optimizer_backends": {"mock": agent.config["optimizer"]["backend_mock"],
                               "real_aspen": agent.config["optimizer"]["backend_real"]},
        "thresholds": {
            "gate": agent.config["gate"],
            "robustness": agent.config["robustness"],
            "constraints": [{k: c[k] for k in ("name", "relation", "threshold",
                                               "scale", "source_id", "source_version")}
                            for c in agent.config["constraints"]],
        },
        "python": platform.python_version(), "library_versions": versions,
        "metrics_summary": metrics,
    }
    with open(os.path.join(save_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def save_figures(results, metrics, save_dir):
    """Figure suite driven ONLY by logged arrays (non-negotiable #2: never
    fabricate/zero-fill/flat-line — panels are skipped if a quantity is absent)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _save(fig, stem):
        """Save PNG + PDF for one figure. The PDF CreationDate is stripped so
        regen_from_log.py yields BYTE-IDENTICAL figures on re-run (the PNG carries
        only a version-stable Software tag, so it is already byte-stable). This is
        what makes figure-layout edits $0 and Aspen-free. Mirrors CS1."""
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, stem + ".png"), dpi=150)
        fig.savefig(os.path.join(save_dir, stem + ".pdf"),
                    metadata={"CreationDate": None})
        plt.close(fig)

    # Colorblind-safe palette (Paul Tol 'bright'): distinguishable under common
    # color-vision deficiency and in grayscale (reinforced with hatching where two
    # adjacent series could still be confused).
    C_GREEN, C_BLUE, C_RED = "#228833", "#4477AA", "#EE6677"
    C_GREY, C_ORANGE, C_PURPLE = "#BBBBBB", "#EE7733", "#AA3377"

    def _bar_labels(ax, xs, ys, fmt="{:.0f}", fs=7):
        """Annotate bar tops with their value (skips NaN/None so absent bars stay
        absent — never fabricates a zero)."""
        for xpos, yval in zip(xs, ys):
            if yval is None or (isinstance(yval, float) and np.isnan(yval)):
                continue
            ax.annotate(fmt.format(yval), (xpos, yval), ha="center", va="bottom",
                        fontsize=fs, xytext=(0, 2), textcoords="offset points")

    order = ["B0_ungated_llm", "B1_llm_rag_nogate", "B2_optimizer_gate", "full"]
    by = lambda s: [r for r in results if r["system"] == s]

    # F1 — optimizer DUTY convergence (full system): best-feasible reboiler duty vs
    # evaluation. Only cases whose RELEASED design came from the duty optimizer are
    # genuine duty-convergence curves; a counterfactual release optimizes DEVIATION
    # from the incumbent (not duty), so plotting its history on a duty axis would be
    # misleading — those cases are excluded and named in an annotation instead.
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted, cf_cases = False, []
    for r in by("full"):
        if r.get("counterfactual_used"):
            cf_cases.append(r["case_id"]); continue
        ts = [t for t in r["time_series"] if t["best_feasible_objective"] is not None]
        if ts:
            ax.plot([t["eval"] for t in ts],
                    [t["best_feasible_objective"] for t in ts],
                    marker=".", ms=4, label=r["case_id"]); plotted = True
    ax.set_xlabel("Aspen evaluation"); ax.set_ylabel("best feasible reboiler duty (kW)")
    ax.set_title("Fig. 1 — Optimizer convergence (full system; GP-BO on real Aspen)")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=8, title="duty-optimized release")
    else:
        ax.text(0.5, 0.5, "no duty-convergence trajectory logged", ha="center",
                transform=ax.transAxes)
    if cf_cases:
        ax.text(0.98, 0.04,
                "released via minimal-deviation counterfactual\n(deviation-optimized, "
                "not a duty search): " + ", ".join(cf_cases) + "\n(released duty in Fig. 6)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                color="#555", bbox=dict(boxstyle="round", fc="#f0f0f0", ec="#bbb"))
    _save(fig, "Fig1_convergence")

    # F2 — among RELEASED designs: nominal-feasible %, robust %, and stale
    # false-release %. Three series so the gate's SPECIFIC value is visible:
    # e.g. ablate_gate releases nominally-feasible but NON-robust designs (and
    # false-releases on stale) — not "infeasible". Systems that abstain everywhere
    # have no released designs to rate (no bar) and are annotated "abstains".
    fig, ax = plt.subplots(figsize=(11, 5))
    sysn = [s for s in (order + ABLATIONS) if s in metrics]
    pct = lambda s, k: (metrics[s].get(k) * 100) if metrics[s].get(k) is not None else np.nan
    nf = [pct(s, "nominal_feasible_rate") for s in sysn]
    rb = [pct(s, "robust_release_rate") for s in sysn]
    fr = [metrics[s]["false_release_rate_stale"] * 100 for s in sysn]
    x = np.arange(len(sysn)); w = 0.26
    ax.bar(x - w, nf, w, label="nominal-feasible % (of released)", color=C_GREEN)
    ax.bar(x,     rb, w, label="robust % (of released)", color=C_BLUE)
    # hatch the "bad" series so it is unmistakable in grayscale / under CVD.
    ax.bar(x + w, fr, w, label="stale false-release %", color=C_RED, hatch="//",
           edgecolor="white")
    for i, s in enumerate(sysn):
        if metrics[s]["n_released"] == 0:
            ax.text(x[i], 3, "abstains", rotation=90, ha="center", va="bottom",
                    fontsize=7, color="#555")
    ax.set_xticks(x); ax.set_xticklabels(sysn, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("% of released designs"); ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Fig. 2 — Released-design quality: nominal-feasible / robust / "
                 "stale false-release")
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    _save(fig, "Fig2_rates")

    # F3 — released reboiler duty per system on the nominal case (logged).
    fig, ax = plt.subplots(figsize=(8, 5))
    duties, labs = [], []
    for s in sysn:
        rs = [r for r in by(s) if r["case_id"] == "C0_nominal" and r["released"]
              and r["objective"] is not None]
        if rs:
            duties.append(rs[0]["objective"]); labs.append(s)
    if duties:
        xs = list(range(len(duties)))
        ax.bar(xs, duties, color=C_BLUE)
        _bar_labels(ax, xs, duties, fmt="{:.0f}", fs=8)
        ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=20,
                                              ha="right", fontsize=8)
        lim = CONFIG["constraints"][3]["threshold"]
        ax.axhline(lim, color=C_RED, ls=":",
                   label=f"utility duty limit ({lim:.0f} kW; slack)")
        ax.set_ylabel("released reboiler duty (kW)")
        ax.set_ylim(0, lim * 1.08); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no released designs on nominal", ha="center",
                transform=ax.transAxes)
    ax.set_title("Fig. 3 — Released reboiler duty (nominal case)")
    _save(fig, "Fig3_duty")

    # F4 — full-system certificate constraint margins per released case (logged).
    fig, ax = plt.subplots(figsize=(9, 5))
    rel = [r for r in by("full") if r["released"] and r.get("certificate")]
    if rel:
        names = list(rel[0]["certificate"]["margins"].keys())
        mat = np.array([[r["certificate"]["margins"].get(n, np.nan) for n in names]
                        for r in rel])
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=2.0)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30,
                                                             ha="right", fontsize=7)
        ax.set_yticks(range(len(rel))); ax.set_yticklabels([r["case_id"] for r in rel],
                                                           fontsize=8)
        for i in range(len(rel)):
            for j in range(len(names)):
                # white text on the saturated dark-green (high-margin) cells, black
                # elsewhere — legible across the whole RdYlGn range.
                tc = "white" if mat[i, j] >= 1.2 else "black"
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color=tc)
        fig.colorbar(im, ax=ax,
                     label="normalized margin (>=0 feasible; color clipped at 2.0)")
    else:
        ax.text(0.5, 0.5, "no released full-system designs", ha="center",
                transform=ax.transAxes)
    ax.set_title("Fig. 4 — Released-design constraint margins (full system)")
    _save(fig, "Fig4_margins")

    # F5 — Aspen Plus evaluation count per system (cost; logged).
    fig, ax = plt.subplots(figsize=(8, 5))
    ev = [metrics[s]["total_aspen_evals"] for s in sysn]
    xs = list(range(len(sysn)))
    ax.bar(xs, ev, color=C_PURPLE)
    ax.set_yscale("log")
    _bar_labels(ax, xs, ev, fmt="{:.0f}", fs=7)
    ax.set_xticks(xs); ax.set_xticklabels(sysn, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("total model evaluations (log scale)")
    ax.set_ylim(top=max(ev) * 1.6)
    ax.set_title("Fig. 5 — Model-evaluation count per system (log scale)")
    _save(fig, "Fig5_evalcount")

    # F6 — TWO-BASELINE energy framing (R5): full vs the nominal starting design
    # AND vs the ungated-LLM (B0) released point, per scenario. Logged data only;
    # missing bars (abstention / no release) are omitted, never zero-filled.
    cs = cross_system(results)
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = [e for e in cs["energy_two_baseline"]
            if any(e[k] is not None for k in ("full_duty_kW", "nominal_baseline_kW",
                                              "b0_released_duty_kW"))]
    if rows:
        x = np.arange(len(rows)); w = 0.27
        colf = lambda key: [(e[key] if e[key] is not None else np.nan) for e in rows]
        b1 = colf("nominal_baseline_kW"); b2 = colf("b0_released_duty_kW")
        b3 = colf("full_duty_kW")
        ax.bar(x - w, b1, w, label="nominal starting design", color=C_GREY)
        ax.bar(x,     b2, w, label="B0 ungated-LLM released", color=C_ORANGE)
        ax.bar(x + w, b3, w, label="full (gated) released", color=C_GREEN)
        _bar_labels(ax, x - w, b1, fmt="{:.0f}", fs=6)
        _bar_labels(ax, x,     b2, fmt="{:.0f}", fs=6)
        _bar_labels(ax, x + w, b3, fmt="{:.0f}", fs=6)
        ax.set_xticks(x); ax.set_xticklabels([e["case_id"] for e in rows], rotation=20,
                                             ha="right", fontsize=8)
        ax.set_ylabel("reboiler duty (kW)")
        ax.set_ylim(top=np.nanmax(b1 + b2 + b3) * 1.12); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no released designs to compare", ha="center",
                transform=ax.transAxes)
    ax.set_title("Fig. 6 — Energy vs two baselines (nominal design & ungated LLM)")
    _save(fig, "Fig6_energy_two_baseline")
    return ["Fig1_convergence", "Fig2_rates", "Fig3_duty", "Fig4_margins",
            "Fig5_evalcount", "Fig6_energy_two_baseline"]


def export_results(results, metrics, agent, cases, save_dir=RESULT_DIR, verbose=True):
    """Metrics JSON + syntheses + manifest + persisted time-series + Aspen traces
    + figures (Phase 7). Everything is driven by logged data."""
    # metrics JSON: strip the heavy time-series + trace (persisted separately).
    clean = [{k: v for k, v in r.items()
              if k not in ("time_series", "aspen_trace")} for r in results]
    with open(os.path.join(save_dir, "metrics_v2.json"), "w") as f:
        json.dump({"per_case": clean, "metrics": metrics,
                   "cross_system": cross_system(results)}, f, indent=2, default=str)

    # LLM syntheses (rationales) for the full-system runs.
    with open(os.path.join(save_dir, "llm_syntheses.txt"), "w", encoding="utf-8") as f:
        for r in results:
            if r["system"] == "full" and r.get("rationale"):
                f.write(f"\n{'='*70}\n{r['case_id']} (model={r['model']}, "
                        f"released={r['released']})\n{'='*70}\n{r['rationale']}\n")

    # persisted per-run time series (npz) + capped Aspen audit traces.
    ts_dir = os.path.join(save_dir, "timeseries"); os.makedirs(ts_dir, exist_ok=True)
    traces = {}
    for r in results:
        tag = f"{r['system']}__{r['case_id']}"
        if r.get("time_series"):
            ts = r["time_series"]
            np.savez(os.path.join(ts_dir, f"{tag}.npz"),
                     eval=np.array([t["eval"] for t in ts]),
                     objective=np.array([(t["objective"] if t["objective"] is not None
                                          else np.nan) for t in ts], float),
                     best_feasible=np.array([(t["best_feasible_objective"]
                                              if t["best_feasible_objective"] is not None
                                              else np.nan) for t in ts], float),
                     min_margin=np.array([t["min_margin"] for t in ts], float))
        traces[tag] = {"total_evals": r.get("aspen_trace_total", 0),
                       "trace": r.get("aspen_trace", [])}
    with open(os.path.join(save_dir, "aspen_traces.json"), "w") as f:
        json.dump(traces, f, indent=2, default=str)

    write_manifest(agent, cases, metrics, save_dir)
    figs = save_figures(results, metrics, save_dir)
    if verbose:
        print(f"  Exported: metrics_v2.json, llm_syntheses.txt, run_manifest.json, "
              f"aspen_traces.json, timeseries/*.npz, {len(figs)} figures, diagnostics.log")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("FlowsheetCopilotR1 — Case Study 2 (fail-closed flowsheet co-pilot)")
    print(f"  MOCK_ASPEN={MOCK_ASPEN}  MOCK_LLM={MOCK_LLM}  BASE_SEED={BASE_SEED}")
    print(f"  system={CONFIG['system']['name']} ({CONFIG['system']['property_method']}), "
          f"objective=min {CONFIG['objective']['name']}")

    cases = build_cases()
    subset = os.environ.get("FCO_SCENARIOS", "").strip()
    if subset:
        keep  = {s.strip() for s in subset.split(",") if s.strip()}
        cases = [c for c in cases if c["id"] in keep]
        print(f"  [subset] FCO_SCENARIOS -> {[c['id'] for c in cases]}")

    flowsheet = AspenFlowsheet(mock=MOCK_ASPEN)
    rag       = RAGEngine()
    optimizer = Optimizer(CONFIG)
    print(f"  Aspen backend: {flowsheet.backend} | RAG dense: {rag.dense_backend}")

    registry = MCPToolRegistry()
    agent    = FlowsheetCopilotAgent(CONFIG, flowsheet, rag, optimizer, registry,
                                     mock_llm=MOCK_LLM)
    registry.register("run_aspen", flowsheet.evaluate, "Run Aspen Plus")
    registry.register("retrieve_knowledge", rag.retrieve, "Dense rationale retrieval")
    registry.register("extract_constraints", rag.retrieve_constraint, "Sparse constraint retrieval")
    registry.register("optimize", optimizer.optimize, "Feasibility-aware optimizer")
    registry.register("evaluate_certificate", agent.certify, "Admissibility certificate")

    # LIVE pre-flight: make the projected spend + the hard ceiling visible before any
    # paid call, and warn if no ceiling is set (recommend $2-3). Mock mode prints nothing.
    if not MOCK_LLM:
        if LLM_COST_ABORT_USD <= 0:
            raise SystemExit("[LIVE][ABORT] FCO_COST_ABORT_USD<=0 — refusing to start a live "
                             "run without a spend ceiling. Set FCO_COST_ABORT_USD (e.g. 2-3).")
        proj = agent.project_live_cost(cases)
        print(f"  [LIVE] projected worst-case ${proj['projected_worstcase_usd']} over "
              f"{proj['n_live_calls']} calls | cost ceiling "
              f"FCO_COST_ABORT_USD=${LLM_COST_ABORT_USD:.2f} | contamination abort "
              f">={int(LLM_CONTAMINATION_ABORT*100)}% | spend restored on resume "
              f"${agent.llm_cost_usd():.4f}")

    results = run_harness(agent, cases)
    metrics = compute_metrics(results)

    print(f"\n  {'system':<24}{'rel':>4}{'abs':>4}{'feas%':>7}{'falseRel':>9}"
          f"{'dE%':>7}{'evals':>7}")
    for s in ("B0_ungated_llm", "B1_llm_rag_nogate", "B2_optimizer_gate", "full",
              *ABLATIONS):
        m = metrics.get(s)
        if not m:
            continue
        fr = "-" if m["feasible_solution_rate"] is None else f"{m['feasible_solution_rate']*100:.0f}"
        de = "-" if m["mean_energy_improvement"] is None else f"{m['mean_energy_improvement']*100:.1f}"
        print(f"  {s:<24}{m['n_released']:>4}{m['n_abstained']:>4}{fr:>7}"
              f"{m['false_release_rate_stale']*100:>8.0f}%{de:>7}{m['total_aspen_evals']:>7}")

    cs = cross_system(results)
    print("\n  Energy — TWO baselines (never a single number):")
    for e in cs["energy_two_baseline"]:
        dn = "-" if e["dE_vs_nominal"] is None else f"{e['dE_vs_nominal']*100:+.1f}%"
        d0 = "-" if e["dE_vs_b0"] is None else f"{e['dE_vs_b0']*100:+.1f}%"
        print(f"    {e['case_id']:<14} full={e['full_duty_kW']} kW | "
              f"vs nominal {dn} | vs B0 {d0}")
    print(f"    mean dE vs nominal={cs['mean_dE_vs_nominal']}  vs B0={cs['mean_dE_vs_b0']}")
    print("\n  full vs B2 (both gated) — robustness + false-release:")
    for fb in cs["full_vs_b2"]:
        print(f"    {fb['case_id']:<14} full[rel={fb['full']['released']} "
              f"rob={fb['full']['robust']} viol={fb['full']['violates_current']}]  "
              f"B2[rel={fb['b2']['released']} rob={fb['b2']['robust']} "
              f"viol={fb['b2']['violates_current']}]")

    export_results(results, metrics, agent, cases)
    print(f"\n  Ran {len(cases)} cases x {len(set(r['system'] for r in results))} systems "
          f"-> {len(results)} runs; outputs at {os.path.abspath(RESULT_DIR)}")


if __name__ == "__main__":
    main()
