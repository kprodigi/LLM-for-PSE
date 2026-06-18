# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 13:01:53 2026

@author: kkhoda
"""

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  AXIOM v2 — Agentic Extraction Intelligence with Optimized Multi-scale      ║
║             Reasoning for Safety-Critical Process Systems                    ║
║                                                                              ║
║  Journal Study: Agentic AI with Multi-Hypothesis Bayesian Belief Tracking,  ║
║  Ensemble Kalman Filtering, and Information-Theoretic Control Action         ║
║  Selection for Abnormal Event Management in Exothermic CSTRs                ║
║                                                                              ║
║  Five Novel Technical Pillars:                                               ║
║  P1. Multi-hypothesis Bayesian belief tracker (posterior over causal faults) ║
║  P2. Ensemble Kalman Filter — uncertainty-aware real-time state estimation   ║
║  P3. Information-theoretic action selection (max expected information gain)  ║
║  P4. Causal attribution via local sensitivity analysis (not do-calculus)     ║
║  P5. Counterfactual trajectory engine for post-incident investigation        ║
║                                                                              ║
║  Comparators: (1) Grid-search controller (oracle fault params),              ║
║               (2) Rule-based SIS, (3) PID-only control                        ║
║                                                                              ║
║  Scenarios: 10 fault scenarios spanning single, compound, cascading,        ║
║             time-varying, and adversarial faults                             ║
║                                                                              ║
║  Dependencies: pip install anthropic scipy numpy scikit-learn matplotlib     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, re, json, time, logging, warnings, copy
from collections import defaultdict
from dataclasses import dataclass, asdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq
from scipy.stats import entropy as kl_entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

warnings.filterwarnings("ignore")

# Vendored kernel: this case study carries its OWN copy of the kernel at
# case1_reactor/core/ (self-contained; no shared top-level core). Repo root on
# sys.path so `case1_reactor.core` resolves when run as a script, via -m, or under
# tests; the two case studies' kernels are independent copies by design.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from case1_reactor.core.env import _env_flag, BASE_SEED
from case1_reactor.core.tools import MCPToolRegistry as _CoreMCPToolRegistry
from case1_reactor.core.diagnostics import make_diagnostics_logger


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# The Anthropic API key is read from the environment — never hardcode it.
# A missing key only raises when a live LLM call is actually attempted
# (see AXIOMAgentV2._client_or_raise), not at import time. Set the env var
# AXIOM_MOCK_LLM=1 to run the full pipeline without any API call/credits.
API_KEY    = os.environ.get("ANTHROPIC_API_KEY")
# AXIOM_LLM_MODEL: per-run override (default = the locked-study pin), mirroring CS2's
# FCO_LLM_MODEL/_WEAK pattern -- makes the pre-registered cross-model arm (A2 grounded@haiku,
# H3 model-independence) runnable with the SAME haiku id as CS2's weak group, so the
# cross-model check is symmetric across both case studies. Read at import; the pre-flight
# pin, the agent default, and every create call all flow from this one constant.
LLM_MODEL  = os.environ.get("AXIOM_LLM_MODEL", "claude-sonnet-4-6")


MOCK_LLM   = _env_flag("AXIOM_MOCK_LLM")

# Fail-closed gate tolerance (Phase 1.3, AUTHOR DECISION): the action PARSED FROM
# THE EXPLANATION and the verified/executed coolant setpoint must agree to within
# TC_TOL kelvin for the certificate's explanation_consistent flag to hold (P5; was
# action_consistent — same tolerance, now governing the narrative not a proposal).
TC_TOL     = 5.0    # K (round 2, item 1: tightened from 15; the proximity check
                    # is now SECONDARY to the recommendation_safe simulation)
SAFE_MARGIN = 5.0   # K (round 2, item 1): require T_run - peak >= SAFE_MARGIN,
                    # both for the executed action and the LLM recommendation

# Robustness gate term (P6): the committed action must keep peak T below T_run across
# an episode-level Monte-Carlo ensemble drawn from the CALIBRATED parameter
# uncertainty. UAf/Ca0f are drawn from the EnKF posterior (their calibrated spread);
# kinetics (k0) are NOT EnKF-calibrated, so k0 is drawn from a documented relative
# uncertainty (Arrhenius rate-constant uncertainty ~10% for a characterized reaction)
# — AUTHOR-CONFIRMABLE, distinct from the calibrated heat-transfer/feed terms.
N_ROBUST            = 64     # ensemble size for the robustness term
K0_REL_SIGMA        = 0.10   # AUTHOR-CONFIRMABLE: relative 1σ on k0 (kinetic uncertainty)
ROBUSTNESS_THRESHOLD = 0.95  # AUTHOR-CONFIRMABLE: require >= this fraction of the
                             # ensemble safe for the action to be certified robust

# Structural robustness (D5: now ON BY DEFAULT, opt-out AXIOM_NO_HARDEN_STRUCTURAL=1). The
# robustness term ALSO requires the committed action to clear T_run by SAFE_MARGIN on a family
# of two-CSTR-in-series shapes (structural / wrong-model-shape mismatch), AND the safety-screened
# controller picks an action that is structurally safe when one exists -- closing the model-relative
# blind spot the single-tank twin left open (the standalone study found up to 8/22 actions breaching
# a 30/70 split at 557 K; ROBUSTNESS_PLAN.md, Axis 2/3). Set AXIOM_NO_HARDEN_STRUCTURAL=1 to
# reproduce the locked single-tank cs1-live-final study byte-for-byte.
HARDEN_STRUCTURAL = not _env_flag("AXIOM_NO_HARDEN_STRUCTURAL")
STRUCT_SPLITS     = (0.3, 0.4, 0.5, 0.6, 0.7)   # stage-1 fractions spanning the dangerous range

# Diagnosis-confidence gate term (P7): the EnKF calibration (fraction of the realized
# T inside the filter's 95% CI) is a standard, scenario-agnostic filter-consistency
# check — it measures whether the MODELLED fault hypotheses can explain the observed
# (T, Ca). A structurally UNOBSERVABLE fault (e.g. S07 flow reduction: q is confounded
# with UAf and is not in the EnKF state) makes the model misspecified, so the CI
# misses the truth and calibration collapses. A well-calibrated filter sits near 0.95;
# DIAGNOSIS_CALIB_THRESHOLD is set well below that so only a genuinely misspecified
# diagnosis (large margin: S07 ~0.54 vs every diagnosable scenario >= 0.90) is flagged
# low-confidence. It governs the DIAGNOSIS narrative, not the action (which is safe).
DIAGNOSIS_CALIB_THRESHOLD = 0.80   # AUTHOR-CONFIRMABLE

# Time-varying fault injection (P8): S09 is a DEVELOPING cooling-coefficient
# degradation (heat-exchanger fouling/scaling). UAf ramps from its pre-onset value to
# UAF_RAMP_END linearly over UAF_RAMP_DURATION minutes starting at the scenario's
# fault_onset. Rate FIXED and documented BEFORE results — not tuned to the outcome.
UAF_RAMP_END      = 0.45    # AUTHOR-CONFIRMABLE: degraded UAf the ramp settles at
UAF_RAMP_DURATION = 10.0    # min (0.95 -> 0.45 over 10 min = 0.05/min, a developing fault)

def make_uaf_ramp(uaf0, uaf_end, onset, ramp_dur):
    """Return UAf(t): constant uaf0 before onset, linear to uaf_end over ramp_dur min,
    then held at uaf_end. A genuine time-varying plant parameter (P8 S09 injection)."""
    def uaf_fn(t):
        if t < onset:
            return float(uaf0)
        if t >= onset + ramp_dur:
            return float(uaf_end)
        return float(uaf0 + (uaf_end - uaf0) * (t - onset) / ramp_dur)
    return uaf_fn


def _two_cstr_peak(cstr, split, y0, Tc, Ca0f, UAf, q_frac, t_sim):
    """Peak temperature over a TWO-CSTR-in-series plant for the hardened structural-robustness
    check (ROBUSTNESS_PLAN.md, Axis 2/3). EVALUATION-ONLY: never the twin; it stresses the committed
    action against a wrong MODEL SHAPE. Equal-intensity split -- stage 1 gets `split` of the volume
    AND the UA, so per-stage cooling intensity matches the single CSTR. Faults (Ca0f, UAf, q_frac)
    and the feed temperature cstr.T0 enter stage 1; stage 2's feed is stage 1's effluent; both
    jackets see Tc. Peak is over BOTH stages."""
    f = float(split)
    V1 = cstr.V * f;   V2 = cstr.V * (1.0 - f)
    UA1 = cstr.UA * f; UA2 = cstr.UA * (1.0 - f)
    Ca0, Ts0 = float(y0[0]), float(y0[1])

    def rhs(t, y):
        Ca1, T1, Ca2, T2 = y
        tau1 = V1 / (cstr.q * q_frac); tau2 = V2 / (cstr.q * q_frac)
        r1 = cstr.k(T1) * Ca1;         r2 = cstr.k(T2) * Ca2
        dCa1 = (Ca0f - Ca1) / tau1 - r1
        dT1 = (cstr.T0 - T1) / tau1 - (cstr.dH / cstr.rho_Cp) * r1 \
            - (UA1 * UAf / (cstr.rho_Cp * V1)) * (T1 - Tc)
        dCa2 = (Ca1 - Ca2) / tau2 - r2
        dT2 = (T1 - T2) / tau2 - (cstr.dH / cstr.rho_Cp) * r2 \
            - (UA2 * UAf / (cstr.rho_Cp * V2)) * (T2 - Tc)
        return [dCa1, dT1, dCa2, dT2]

    # Terminal event: stop as soon as either stage crosses a cap above the runaway limit.
    # The structural check only needs the BOOLEAN "does the peak breach T_run", so there is no
    # need to integrate the (stiff, slow) full runaway blow-up -- this keeps the per-action
    # structural screen cheap enough to run inside the gate and the controller (D5).
    _cap = float(cstr.T_run) + 25.0

    def _hit_cap(t, y):
        return max(y[1], y[3]) - _cap
    _hit_cap.terminal = True
    _hit_cap.direction = 1

    sol = solve_ivp(rhs, (0, t_sim), [Ca0, Ts0, Ca0, Ts0], method="LSODA",
                    t_eval=np.linspace(0, t_sim, 100), events=_hit_cap, rtol=1e-6, atol=1e-8)
    if not sol.success:
        return float("inf")
    peak = max(np.max(sol.y[1]), np.max(sol.y[3]))
    if sol.t_events is not None and len(sol.t_events) and len(sol.t_events[0]):
        peak = max(peak, _cap)        # crossed the cap -> definitively breaches T_run
    return float(peak)


# Runaway Risk Index bounds (round 2, item 9): RRI = heat_gen / heat_removal is
# bounded to [0, RRI_MAX]; when heat removal falls to/below REM_FLOOR (the jacket
# is net-heating) the regime is flagged "removal-limited" and RRI is capped,
# preserving SOP-014 threshold semantics (1.0/1.5/1.8) without the spurious ~1e5
# values the old max(rem, 1.0) clamp produced.
RRI_MAX    = 5.0    # ceiling, well above SOP-014's 1.8 emergency threshold
REM_FLOOR  = 1.0    # cal/min; heat-removal floor

# Results live next to this module (case1_reactor/results/), so the output
# location is independent of the working directory (repo reorg; was
# "./axiom_v2_results" at the old repo root).
# Override with AXIOM_RESULT_DIR so each ablation arm of the re-run writes to a SEPARATE dir
# (run_live_eval._tracked_dir redirects the tracked copies too), so the locked study in
# live_results/ is NEVER overwritten. Set the env BEFORE importing this module.
RESULT_DIR = (os.environ.get("AXIOM_RESULT_DIR") or
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
os.makedirs(RESULT_DIR, exist_ok=True)

# Reproducibility (Phase 4.1): determinism comes from EXPLICIT per-scenario
# numpy Generators (np.random.default_rng), not a single global seed whose
# outcome depends on execution order. The legacy global stream is still seeded
# as a harmless safety net, but the scenario pipeline draws only from the
# per-scenario Generator derived from BASE_SEED and the scenario id.
np.random.seed(BASE_SEED)            # BASE_SEED imported from case1_reactor.core.env (vendored, 42)


def scenario_seed(sid, base=BASE_SEED):
    """Deterministic per-scenario seed derived from the base seed and the
    numeric part of the scenario id, so a scenario is reproducible regardless
    of order or which subset is run."""
    digits = "".join(ch for ch in str(sid) if ch.isdigit())
    return base * 1000 + (int(digits) if digits else 0)


def runaway_risk_index(cstr, T, Ca, Tc, UAf):
    """
    Bounded Runaway Risk Index (round 2, item 9): RRI = heat_gen / heat_removal,
    capped at RRI_MAX. Returns (rri_array, removal_limited_array).

    heat_gen = |dH|*k(T)*Ca*V ; heat_removal = UA*UAf*(T-Tc) + q*rhoCp*(T-T0).
    When heat_removal <= REM_FLOOR (the jacket is net-heating) the regime is
    flagged removal-limited; RRI is capped via max(rem, REM_FLOOR) and the
    overall min(..., RRI_MAX), so the metric stays in [0, RRI_MAX] and the SOP
    thresholds (1.0/1.5/1.8) remain meaningful without the old artifact.
    """
    T  = np.asarray(T, float); Ca = np.asarray(Ca, float); Tc = np.asarray(Tc, float)
    r   = cstr.k(T) * Ca
    gen = np.abs(cstr.dH) * r * cstr.V
    rem = cstr.UA * UAf * (T - Tc) + cstr.q * cstr.rho_Cp * (T - cstr.T0)
    removal_limited = rem <= REM_FLOOR
    rri = np.minimum(gen / np.maximum(rem, REM_FLOOR), RRI_MAX)
    return rri, removal_limited

# Diagnostics log (Phase 2.5): integration/EIG/scorer failures are recorded
# here instead of being silently swallowed by bare excepts.
diag_log = make_diagnostics_logger("axiom.diagnostics",
                                   os.path.join(RESULT_DIR, "diagnostics.log"),
                                   logging.WARNING)

plt.rcParams.update({
    "font.family":"Arial", "font.size":11,
    "axes.linewidth":1.2, "axes.spines.top":False, "axes.spines.right":False,
    "axes.grid":True, "grid.alpha":0.25, "grid.linewidth":0.5,
    "legend.frameon":False, "figure.dpi":150,
    "savefig.dpi":300, "savefig.bbox":"tight",
})

C = {  # color palette
    "axiom":"#1F3864", "gs_oracle":"#C45911", "gs_fair":"#00897B",
    "sis":"#2E7D32", "pid":"#6A1B9A",
    "safe":"#43A047", "warning":"#FB8C00", "critical":"#E53935",
    "neutral":"#78909C", "info":"#1565C0",
    "h1":"#1565C0","h2":"#6A1B9A","h3":"#B71C1C","h4":"#E65100",
    "h5":"#1B5E20","h6":"#37474F","h7":"#880E4F","h8":"#4E342E",
}
SC_COLORS = [C["h1"],C["h2"],C["h3"],C["h4"],C["h5"],
             C["h6"],C["h7"],C["h8"],"#0277BD","#558B2F"]

# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 — EXOTHERMIC CSTR PROCESS MODEL (enhanced)
# ══════════════════════════════════════════════════════════════════════════════

class CSTRModel:
    """
    Van de Vusse-inspired exothermic CSTR with:
    - Arrhenius kinetics (A→B first order, exothermic)
    - Jacket heat removal with degradable UA
    - Time-varying fault injection API
    - This is the Seborg / Edgar & Mellichamp controllable exothermic-CSTR
      benchmark (q, V, Ca0, T0, Tc, E/R, rho_Cp, UA all match it), re-parameterized
      to be genuinely runaway-capable yet controllable: |dH|=75000 cal/mol
      (adiabatic rise ~|dH|*Ca0/rho_Cp ≈ 150 K) with k0=7.2e9 (10x slower than
      Seborg, so the coolant actuator can keep pace). The original |dH|=5960 was a
      ~10x reduction from Seborg's -5e4 that gave only a ~12 K rise and could never
      reach T_run. At nominal cooling the energy balance has a SINGLE stable low-T
      steady state (no ignition/extinction multiplicity); a cooling-loss (UA decay)
      or feed surge drives a CONTROLLABLE fault-induced runaway — insufficient
      cooling crosses T_run while aggressive cooling recovers — so safe AND unsafe
      recovery actions genuinely both exist. No fold/Hopf at nominal (single-state
      reframe, decision 1). Basis: van Heerden (1953); Bequette; Seborg, Edgar &
      Mellichamp; Stoessel (~150 K rise typical of hazardous exotherms). NOTES.md.
      Safety is guaranteed RELATIVE TO THIS MODEL + the encoded limits, not a plant.
    """
    def __init__(self):
        # Kinetics
        self.k0    = 7.2e9        # pre-exponential (1/min); 10x slower than the
                                  # Seborg value (7.2e10) so the coolant actuator
                                  # can control the runaway (see class docstring)
        self.Ea_R  = 8750.0       # Ea/R (K)
        self.dH    = -75000.0     # heat of reaction (cal/mol); ~1.5x Seborg's
                                  # -5e4 (the original -5960 was a ~10x error),
                                  # giving adiabatic rise ΔT_ad ≈ 150 K
        self.rho_Cp= 500.0        # density x Cp (cal/L/K)
        # Operating
        self.q     = 100.0        # volumetric flow (L/min)
        self.V     = 100.0        # reactor volume (L)
        self.Ca0   = 1.0          # nominal feed (mol/L)
        self.T0    = 350.0        # feed temperature (K)
        self.Tc_nom= 300.0        # nominal coolant (K)
        self.UA    = 5.0e4        # heat transfer coeff (cal/min/K)
        # Setpoints & limits
        # T_sp is the NOMINAL operating temperature = the lower stable steady state
        # of the reparameterized model (lower_stable_ss() -> 326.19 K). Reconciled
        # from the prior 390 K, which no longer corresponds to any steady state.
        self.T_sp  = 326.2
        self.T_warn= 440.0
        self.T_alarm=450.0
        self.T_run = 475.0
        self.T_shut= 485.0
        self.Ca_low= 0.10

    def k(self, T):
        return self.k0 * np.exp(-self.Ea_R / np.clip(T, 200, 1000))

    def odes(self, t, y, Tc, Ca0f, UAf=1.0, q_frac=1.0, noise_T=0.0):
        Ca, T = y[0], y[1]
        r     = self.k(T) * Ca
        tau   = self.V / (self.q * q_frac)
        UA_eff= self.UA * UAf
        dCa   = (Ca0f - Ca)/tau - r
        dT    = (self.T0 - T)/tau \
                - (self.dH/self.rho_Cp)*r \
                - (UA_eff/(self.rho_Cp*self.V))*(T - Tc) \
                + noise_T
        return [dCa, dT]

    def simulate(self, t_span, y0, Tc=None, Ca0f=None,
                 UAf=1.0, q_frac=1.0, noise=0.0, rng=None, n_eval=2):
        Tc   = Tc   if Tc   is not None else self.Tc_nom
        Ca0f = Ca0f if Ca0f is not None else self.Ca0
        if noise and noise > 0:
            # Noisy path: needs a dense grid to index the noise sequence.
            tev = np.linspace(*t_span, 400)
            gen = rng if rng is not None else np.random
            nv  = gen.normal(0, noise, len(tev))
            f   = lambda t, y: self.odes(t, y, Tc, Ca0f, UAf, q_frac,
                                         nv[min(np.searchsorted(tev, t), len(nv)-1)])
            teval = tev
        else:
            # Deterministic path (the only one the agent uses): draw NO random
            # numbers (so it cannot perturb the per-scenario RNG stream — Phase
            # 4.1) and integrate to the endpoints only, since every caller reads
            # just .y[:, -1] (Phase 4.6 performance; t_eval does not affect the
            # integrated endpoint, only which points are interpolated out).
            f     = lambda t, y: self.odes(t, y, Tc, Ca0f, UAf, q_frac, 0.0)
            teval = np.linspace(*t_span, n_eval)
        # Stiff-capable solver (LSODA auto-switches Adams/BDF) with consistent
        # rtol/atol and a max_step cap so fast near-runaway transients are not
        # stepped over (Phase 3.1).
        sol  = solve_ivp(f, t_span, y0, method="LSODA", t_eval=teval,
                         rtol=1e-6, atol=1e-9, max_step=1.0)
        if not sol.success:
            diag_log.warning("simulate: solver failed (t_span=%s y0=%s Tc=%.1f "
                             "UAf=%.2f Ca0f=%.2f): %s", t_span, list(y0), Tc,
                             UAf, Ca0f, sol.message)
        return sol

    def simulate_cl(self, t_total, y0, ctrl_fn, Ca0f=None,
                    UAf=1.0, q_frac=1.0, noise=0.0, dt=0.25, rng=None, uaf_fn=None):
        """Closed-loop simulation with time-varying control. If uaf_fn is given, the
        plant's UAf is TIME-VARYING (uaf_fn(t)) — the P8 developing-fault injection;
        otherwise UAf is the constant passed in."""
        Ca0f  = Ca0f if Ca0f is not None else self.Ca0
        gen   = rng if rng is not None else np.random       # explicit RNG (4.1)
        ts,Ts,Cas,Tcs,qs=[0.],[y0[1]],[y0[0]],[self.Tc_nom],[q_frac]
        y = list(y0)
        n_fail = 0
        for step in range(int(t_total/dt)):
            t_now = step*dt
            ctrl  = ctrl_fn(t_now, y[1], y[0])
            Tc_now= ctrl.get("Tc", self.Tc_nom)
            qf_now= ctrl.get("q_frac", q_frac)
            uaf_now = uaf_fn(t_now) if uaf_fn is not None else UAf   # P8: time-varying plant
            nt    = gen.normal(0, noise) if noise else 0.0
            sol   = solve_ivp(
                lambda t,yy: self.odes(t,yy,Tc_now,Ca0f,uaf_now,qf_now,nt),
                (t_now, t_now+dt), y, method="LSODA",
                rtol=1e-6, atol=1e-9, max_step=dt)
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                y = [sol.y[0,-1], sol.y[1,-1]]   # accept only a valid integration
            else:
                n_fail += 1
                diag_log.warning("simulate_cl: step %d (t=%.2f) integration "
                                 "failed (%s); holding previous state", step,
                                 t_now, getattr(sol, "message", "non-finite"))
            ts.append(t_now+dt); Cas.append(y[0])
            Ts.append(y[1]);     Tcs.append(Tc_now); qs.append(qf_now)
        if n_fail:
            diag_log.warning("simulate_cl: %d/%d steps failed to integrate",
                             n_fail, int(t_total/dt))
        return (np.array(ts), np.array(Cas),
                np.array(Ts),  np.array(Tcs), np.array(qs))

    def steady_states(self, Tc=None, T_lo=300.0, T_hi=540.0, n_scan=600):
        """
        Locate ALL steady states by solving the 1-D steady-state residual
        g(T)=0 with bracketed root finding, then classify each by linearized
        stability. At the nominal parameters this returns a SINGLE stable state
        (the small adiabatic rise precludes multiplicity); the method would
        surface any unstable/middle branch if the parameters produced one, but
        none exists here (single-state reframe, decision 1).

        At steady state dCa/dt=0 gives Ca(T) = Ca0 / (1 + k(T)*tau), which is
        substituted into dT/dt=0 to form g(T). Bracketed root finding (unlike
        forward integration, which can only land on stable branches) would
        recover an unstable saddle if one were present.

        Returns a list of dicts {Ca, T, stable, eigs} sorted by T ascending.
        """
        Tc  = Tc if Tc is not None else self.Tc_nom
        tau = self.V / self.q

        def Ca_of_T(T):
            return self.Ca0 / (1.0 + self.k(T) * tau)

        def g(T):
            Ca = Ca_of_T(T)
            return ((self.T0 - T) / tau
                    - (self.dH / self.rho_Cp) * self.k(T) * Ca
                    - (self.UA / (self.rho_Cp * self.V)) * (T - Tc))

        Tg   = np.linspace(T_lo, T_hi, n_scan)
        vals = np.array([g(T) for T in Tg])
        roots = []
        for i in range(len(Tg) - 1):
            if vals[i] == 0.0:
                roots.append(float(Tg[i]))
            elif vals[i] * vals[i + 1] < 0:
                try:
                    roots.append(float(brentq(g, Tg[i], Tg[i + 1], xtol=1e-8)))
                except Exception as err:
                    diag_log.warning("steady_states: brentq failed on "
                                     "[%.2f, %.2f]: %s", Tg[i], Tg[i + 1], err)
        out = []
        for T in sorted(roots):
            if out and abs(T - out[-1]["T"]) < 1e-3:
                continue
            Ca = Ca_of_T(T)
            _, eigs, stable = self.linearize(Ca, T, Tc)
            out.append({"Ca": float(Ca), "T": float(T), "stable": bool(stable),
                        "eigs": [complex(e) for e in eigs]})
        return out

    def lower_stable_ss(self, Tc=None):
        """
        The lowest-temperature STABLE steady state, selected explicitly (by
        sorting on T and checking stability), not by positional index.
        Falls back to the lowest-T state if none are classified stable.
        """
        ss = self.steady_states(Tc)
        stable = [s for s in ss if s["stable"]]
        if stable:
            return sorted(stable, key=lambda s: s["T"])[0]
        if ss:
            return sorted(ss, key=lambda s: s["T"])[0]
        return {"Ca": 0.5, "T": 350.0, "stable": False, "eigs": []}

    def linearize(self, Ca_ss, T_ss, Tc=None):
        """Compute Jacobian A at steady state for stability analysis."""
        Tc   = Tc if Tc is not None else self.Tc_nom
        eps  = 1e-5
        y_ss = [Ca_ss, T_ss]
        f0   = self.odes(0, y_ss, Tc, self.Ca0)
        A    = np.zeros((2,2))
        for j in range(2):
            yp      = list(y_ss); yp[j] += eps
            fp      = self.odes(0, yp, Tc, self.Ca0)
            A[:,j]  = [(fp[i]-f0[i])/eps for i in range(2)]
        eigs = np.linalg.eigvals(A)
        stable = all(np.real(e) < 0 for e in eigs)
        return A, eigs, stable

    def optimize_action(self, T, Ca, Ca0f=None, UAf=1.0, q_frac=1.0,
                         horizon=5.0, use_flow=False):
        """
        Multi-variable receding horizon optimizer.
        Optimizes Tc (and optionally q_frac) over 'horizon' minutes.
        Objective: minimize (T_final - T_sp)^2 + control effort penalty.
        """
        Ca0f = Ca0f if Ca0f is not None else self.Ca0
        def obj(u):
            Tc_opt  = u[0]
            qf_opt  = u[1] if use_flow else q_frac
            sol     = self.simulate((0,horizon),[Ca,T],
                                    Tc=Tc_opt, Ca0f=Ca0f,
                                    UAf=UAf, q_frac=qf_opt)
            T_f     = sol.y[1,-1]
            pen_Tc  = 0.005*(Tc_opt - self.Tc_nom)**2
            pen_q   = 0.1*(qf_opt - q_frac)**2 if use_flow else 0
            return (T_f - self.T_sp)**2 + pen_Tc + pen_q

        x0 = [self.Tc_nom, q_frac] if use_flow else [self.Tc_nom]
        bounds = [(250,350),(0.5,1.0)] if use_flow else [(250,350)]
        res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")

        Tc_opt = res.x[0]
        qf_opt = res.x[1] if use_flow else q_frac
        sol_opt= self.simulate((0,horizon),[Ca,T],
                               Tc=Tc_opt, Ca0f=Ca0f, UAf=UAf, q_frac=qf_opt)
        return {
            "Tc":   round(Tc_opt, 2),
            "q_frac": round(qf_opt, 3),
            "T_pred": round(sol_opt.y[1,-1], 2),
            "Ca_pred":round(sol_opt.y[0,-1], 4),
            "success":res.success,
            "obj_val":round(float(res.fun), 4),
        }

    def detect(self, T, Ca, UAf=1.0, dT_dt=None):
        """
        Rich anomaly detection returning events, severity, and
        quantitative risk metrics used by the Bayesian hypothesis tracker.
        """
        events   = []
        severity = "normal"
        margin   = self.T_run - T
        risk_idx = max(0.0, (T - self.T_alarm) / (self.T_run - self.T_alarm))

        if T >= self.T_shut:
            events.append("emergency_shutdown_required"); severity="shutdown"
        elif T >= self.T_run:
            events.append("thermal_runaway_imminent");    severity="critical"
        elif T >= self.T_alarm:
            events.append("high_temperature_alarm");      severity="warning"
        elif T >= self.T_warn:
            events.append("elevated_temperature");         severity="caution"

        if Ca <= self.Ca_low:
            events.append("near_complete_conversion")
        if UAf < 0.60:
            events.append("heat_transfer_degradation")
        if UAf < 0.40:
            events.append("severe_cooling_loss")
        if dT_dt is not None and dT_dt > 5.0:
            events.append("rapid_temperature_rise")
        if dT_dt is not None and dT_dt < -8.0:
            events.append("rapid_temperature_drop")

        return {
            "T":T, "Ca":Ca, "UAf":UAf, "events":events,
            "severity":severity, "margin":round(margin,2),
            "risk_index":round(risk_idx,4),
            "dT_dt": round(dT_dt,3) if dT_dt is not None else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 — ENSEMBLE KALMAN FILTER STATE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleKalmanFilter:
    """
    EnKF for real-time state estimation of the CSTR under:
    - Measurement noise on T (thermocouple) and Ca (inline analyzer)
    - Model uncertainty (UAf and Ca0f unknown, treated as augmented state)
    - Ensemble size N for uncertainty quantification

    State vector: [Ca, T, UAf_est, Ca0f_est]  (augmented)
    Observations: [T_meas, Ca_meas]
    """
    def __init__(self, cstr, N=80, rng=None):
        self.cstr = cstr
        self.N    = N          # ensemble size
        # Explicit Generator (Phase 4.1): all EnKF randomness draws from self.rng.
        self.rng  = rng if rng is not None else np.random.default_rng()

        # Measurement-noise covariance R, made consistent with the noise the
        # observations are actually corrupted by in axiom_ctrl_fn (T: 1.5 K std,
        # Ca: 0.01 std). Previously R implied 2 K / 0.0316 std, biasing any
        # calibration claim (Phase 2.6). No deliberate inflation is applied.
        self.obs_sigma = np.array([1.5, 0.01])          # [T, Ca] obs noise std
        self.R = np.diag(self.obs_sigma ** 2)           # diag([2.25, 1e-4])

        # Process noise (model uncertainty)
        self.Q_diag = np.array([0.0005, 0.5, 0.0002, 0.0002])

        # Initialize ensemble around nominal
        self.ensemble = self._init_ensemble([0.5, 350.0, 1.0, 1.0])
        self.history  = []     # (t, mean, std, obs)

    def _init_ensemble(self, x0):
        """Initialize N particles with small perturbations."""
        ens   = np.tile(x0, (self.N, 1))
        noise = self.rng.standard_normal((self.N, 4)) * np.array([0.05,5.0,0.05,0.05])
        ens  += noise
        ens[:,0] = np.clip(ens[:,0], 0.001, 1.5)    # Ca
        ens[:,1] = np.clip(ens[:,1], 280,   600)     # T
        ens[:,2] = np.clip(ens[:,2], 0.1,   1.5)     # UAf
        ens[:,3] = np.clip(ens[:,3], 0.1,   2.0)     # Ca0f
        return ens

    def screen_params(self, ua_pct=5, ca0_pct=95):
        """
        Conservative (worst-case) fault parameters from the current ensemble for
        the safety screen (Phase 2.1): a low UAf percentile (worst cooling)
        and a high Ca0f percentile (most heat generation). This makes the safety
        filter reflect the degraded plant rather than nominal cooling/feed.
        """
        return (float(np.percentile(self.ensemble[:, 2], ua_pct)),
                float(np.percentile(self.ensemble[:, 3], ca0_pct)))

    def predict(self, Tc, dt=0.25):
        """Propagate ensemble forward by dt using CSTR ODEs."""
        new_ens = np.zeros_like(self.ensemble)
        for i in range(self.N):
            Ca, T, UAf, Ca0f = self.ensemble[i]
            try:
                sol = self.cstr.simulate(
                    (0, dt), [Ca, T], Tc=Tc, Ca0f=Ca0f,
                    UAf=UAf, noise=0.0)
                if not sol.success:
                    raise RuntimeError(sol.message)
                new_ens[i, :2] = [sol.y[0,-1], sol.y[1,-1]]
            except Exception as err:
                diag_log.warning("EnKF.predict: particle %d integration failed "
                                 "(Ca=%.3f T=%.1f UAf=%.2f): %s", i, Ca, T, UAf, err)
                new_ens[i, :2] = [Ca, T]
            # Augmented state: add small random walk
            new_ens[i, 2] = np.clip(
                UAf   + self.rng.standard_normal()*0.01, 0.1, 1.5)
            new_ens[i, 3] = np.clip(
                Ca0f  + self.rng.standard_normal()*0.01, 0.1, 2.0)
            # Add process noise
            new_ens[i] += self.rng.standard_normal(4) * np.sqrt(self.Q_diag)
        self.ensemble = new_ens

    def update(self, z_obs, t=None):
        """
        EnKF analysis step.
        z_obs = [T_measured, Ca_measured]
        Returns posterior mean and covariance.
        """
        # Observation operator H: extracts [Ca, T] from state [Ca,T,UAf,Ca0f]
        H     = np.array([[0,1,0,0],[1,0,0,0]], dtype=float)  # [T, Ca]
        N     = self.N
        ens   = self.ensemble                          # (N, 4)

        # Ensemble mean and anomalies
        x_bar = ens.mean(axis=0)                       # (4,)
        A     = ens - x_bar                            # (N, 4)

        # Innovation ensemble
        Y     = (H @ ens.T).T                          # (N, 2)
        y_bar = Y.mean(axis=0)                         # (2,)
        B     = Y - y_bar                              # (N, 2)

        # Ensemble covariance in obs space + noise
        C_yy  = (B.T @ B) / (N-1) + self.R            # (2,2)
        C_xy  = (A.T @ B) / (N-1)                     # (4,2)

        # Kalman gain
        K     = C_xy @ np.linalg.pinv(C_yy)           # (4,2)

        # Perturbed observations
        eps   = self.rng.multivariate_normal([0,0], self.R, N)   # (N,2)
        z_ens = z_obs + eps                            # (N,2)

        # Update ensemble
        innov = z_ens - Y                              # (N,2)
        self.ensemble = ens + (K @ innov.T).T

        # Clip physical bounds
        self.ensemble[:,0] = np.clip(self.ensemble[:,0], 0.001, 1.8)
        self.ensemble[:,1] = np.clip(self.ensemble[:,1], 280,   600)
        self.ensemble[:,2] = np.clip(self.ensemble[:,2], 0.05,  1.5)
        self.ensemble[:,3] = np.clip(self.ensemble[:,3], 0.1,   2.5)

        mean = self.ensemble.mean(axis=0)
        std  = self.ensemble.std(axis=0)
        cov  = np.cov(self.ensemble.T)

        if t is not None:
            self.history.append({"t":t, "mean":mean.copy(), "std":std.copy(),
                                  "obs":z_obs.copy(), "innov_norm":
                                  float(np.linalg.norm(z_obs - y_bar))})
        return mean, std, cov

    def get_state(self):
        mean = self.ensemble.mean(axis=0)
        std  = self.ensemble.std(axis=0)
        return {
            "Ca":    round(float(mean[0]),4),
            "T":     round(float(mean[1]),2),
            "UAf":   round(float(mean[2]),4),
            "Ca0f":  round(float(mean[3]),4),
            "std_T": round(float(std[1]),3),
            "std_Ca":round(float(std[0]),4),
            "std_UA":round(float(std[2]),4),
            "ensemble_spread_T": round(float(np.percentile(
                self.ensemble[:,1],97.5)-np.percentile(
                self.ensemble[:,1],2.5)),2),
        }

    def compute_calibration(self, true_T_series, obs_T_series=None):
        """
        Calibration: fraction of true T values that fall within the EnKF 95% CI
        (mean ± 1.96σ) at the corresponding logged step. Well-calibrated ≈ 0.95.
        Now consistent because R matches the injected observation noise (2.6).
        true_T_series is aligned with self.history from the start of the run.
        """
        hits = 0
        n    = min(len(self.history), len(true_T_series))
        for entry, T_true in zip(self.history[:n], true_T_series[:n]):
            lo = entry["mean"][1] - 1.96*entry["std"][1]
            hi = entry["mean"][1] + 1.96*entry["std"][1]
            if lo <= T_true <= hi:
                hits += 1
        return hits / n if n > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 3 — MULTI-HYPOTHESIS BAYESIAN BELIEF TRACKER
# ══════════════════════════════════════════════════════════════════════════════

FAULT_HYPOTHESES = {
    "H1_normal":            {"desc":"Normal operation — no fault",
                              "UA_range":(0.85,1.15), "Ca0_range":(0.9,1.1),
                              "q_range":(0.9,1.1),    "sensor_ok":True,  "prior":0.30},
    "H2_cooling_partial":   {"desc":"Partial cooling loss (UA 40-65%)",
                              "UA_range":(0.40,0.65), "Ca0_range":(0.9,1.1),
                              "q_range":(0.9,1.1),    "sensor_ok":True,  "prior":0.15},
    "H3_cooling_severe":    {"desc":"Severe cooling loss (UA < 40%)",
                              "UA_range":(0.10,0.40), "Ca0_range":(0.9,1.1),
                              "q_range":(0.9,1.1),    "sensor_ok":True,  "prior":0.08},
    "H4_feed_spike":        {"desc":"Feed concentration spike (Ca0 > 1.3)",
                              "UA_range":(0.85,1.15), "Ca0_range":(1.3,2.2),
                              "q_range":(0.9,1.1),    "sensor_ok":True,  "prior":0.12},
    "H5_compound":          {"desc":"Compound: partial cooling + feed surge",
                              "UA_range":(0.40,0.70), "Ca0_range":(1.3,2.0),
                              "q_range":(0.9,1.1),    "sensor_ok":True,  "prior":0.08},
    "H6_sensor_drift":      {"desc":"Thermocouple drift (measured T unreliable)",
                              "UA_range":(0.85,1.15), "Ca0_range":(0.9,1.1),
                              "q_range":(0.9,1.1),    "sensor_ok":False, "prior":0.10},
    "H7_cooling_incipient": {"desc":"Incipient cooling degradation (UA 75-90%, mild T rise)",
                              "UA_range":(0.75,0.90), "Ca0_range":(0.9,1.1),
                              "q_range":(0.9,1.1),    "sensor_ok":True,  "prior":0.07},
    "H8_flow_loss":         {"desc":"Feed flow reduction (q < 70% nominal)",
                              "UA_range":(0.85,1.15), "Ca0_range":(0.9,1.1),
                              "q_range":(0.40,0.70),  "sensor_ok":True,  "prior":0.05},
    "H9_runaway_onset":     {"desc":"Incipient thermal runaway (all factors)",
                              "UA_range":(0.10,0.55), "Ca0_range":(1.2,2.0),
                              "q_range":(0.7,1.0),    "sensor_ok":True,  "prior":0.03},
    "H10_cascade":          {"desc":"Cascading failure (cooling + sensor + flow)",
                              "UA_range":(0.10,0.50), "Ca0_range":(0.9,1.2),
                              "q_range":(0.4,0.8),    "sensor_ok":False, "prior":0.02},
}


class HypothesisBelief:
    """
    Multi-hypothesis Bayesian belief tracker.

    Maintains P(H_i | observations) for all competing causal hypotheses.
    At each observation step, computes the likelihood P(obs | H_i) using the
    EnKF state estimate, updates posteriors via Bayes rule, and triggers
    hypothesis revision when KL-divergence between consecutive posteriors
    exceeds a calibrated threshold.

    This is the core technical novelty of AXIOM: the agent does not just
    detect "high temperature alarm" — it maintains a probability distribution
    over WHY the alarm is occurring and designs actions to discriminate between
    competing causal explanations.
    """

    def __init__(self, cstr, kl_threshold=0.15, temper=1.0, prior=None):
        self.cstr         = cstr
        self.kl_threshold = kl_threshold
        # Likelihood tempering / forgetting factor (Phase 3.3): posterior ∝
        # (likelihood ** temper) × prior. temper=1.0 reproduces the original
        # multiplicative update exactly; temper<1.0 down-weights each new
        # observation to curb overconfidence in non-stationary faults. Default
        # is the original behavior (no validated tuned value yet).
        self.temper       = temper
        self.hypotheses   = list(FAULT_HYPOTHESES.keys())
        self.n_hyp        = len(self.hypotheses)
        # prior: optional {hyp_key: weight>=0} from the LLM decision-spec's
        # hypothesis_weighting (P3). It SEEDS the initial posterior; the tracker
        # still updates it with data, so a bad/grounded prior is corrected (the
        # prior washes out — finding 4). None -> the static FAULT_HYPOTHESES priors.
        self.prior_spec   = prior
        self.posterior    = self._initial_posterior(prior)
        self.prior_hist   = [self.posterior.copy()]
        self.kl_hist      = []
        self.revision_events = []
        self.step         = 0

    def _initial_posterior(self, prior=None):
        """Build the normalized initial belief. With a decision-spec weighting,
        specified hypotheses get their weight and the rest a small floor (so none
        is ruled out a priori — a grounded prior only RE-WEIGHTS, it cannot make a
        fault impossible). Without one, the static FAULT_HYPOTHESES priors."""
        if prior is None:
            w = np.array([FAULT_HYPOTHESES[h]["prior"] for h in self.hypotheses], float)
        else:
            floor = 1e-3
            w = np.array([max(float(prior.get(h, 0.0)), floor)
                          for h in self.hypotheses], float)
        s = float(w.sum())
        return (w / s) if s > 0 else np.full(self.n_hyp, 1.0 / self.n_hyp)

    def likelihood(self, obs_T, obs_Ca, enkf_state, h_key):
        """
        P(observation | hypothesis H_i).

        APPROXIMATION (documented honestly): this is a HAND-TUNED heuristic, not
        a calibrated probabilistic likelihood. It multiplies a uniform-range
        consistency score for (UA, Ca0), a sensor-validity score, and a Gaussian
        temperature-alignment score with hand-chosen centers/widths. Combined
        with the multiplicative cumulative update (no transition model), it can
        become overconfident in non-stationary faults — see the optional
        `temper` factor on the belief update.

        Uses the EnKF-estimated UA and Ca0 to score how consistent the
        current system state is with each hypothesis's parameter ranges.
        Also uses T and Ca observations against hypothesis-predicted
        steady-state ranges.
        """
        h       = FAULT_HYPOTHESES[h_key]
        ua_est  = enkf_state["UAf"]
        ca0_est = enkf_state["Ca0f"]

        # Score 1: parameter range consistency (uniform likelihood over range)
        ua_lo, ua_hi   = h["UA_range"]
        ca0_lo,ca0_hi  = h["Ca0_range"]
        ua_score  = 1.0 if ua_lo <= ua_est <= ua_hi else \
                    np.exp(-10*min(abs(ua_est-ua_lo), abs(ua_est-ua_hi))**2)
        ca0_score = 1.0 if ca0_lo <= ca0_est <= ca0_hi else \
                    np.exp(-10*min(abs(ca0_est-ca0_lo),abs(ca0_est-ca0_hi))**2)

        # Score 2: sensor validity check
        T_spread = enkf_state.get("ensemble_spread_T", 5.0)
        if not h["sensor_ok"] and T_spread > 8.0:
            sensor_score = 2.0   # sensor drift hypothesis more likely if spread large
        elif h["sensor_ok"] and T_spread > 8.0:
            sensor_score = 0.3
        else:
            sensor_score = 1.0

        # Score 3: temperature-hypothesis alignment
        T    = obs_T
        cstr = self.cstr
        if h_key == "H1_normal":
            T_score = np.exp(-((T-cstr.T_sp)/20)**2)
        elif h_key in ("H2_cooling_partial","H3_cooling_severe"):
            T_score = np.exp(-((T-460)/20)**2) if T > cstr.T_alarm else 0.3
        elif h_key == "H4_feed_spike":
            T_score = np.exp(-((T-455)/25)**2) if T > cstr.T_warn else 0.2
        elif h_key == "H5_compound":
            T_score = np.exp(-((T-465)/20)**2) if T > cstr.T_alarm else 0.2
        elif h_key == "H7_cooling_incipient":
            # Incipient cooling: the system is BISTABLE, so mild UA loss does NOT
            # produce a "mildly elevated plateau" — it stays on the low branch
            # (UAf 0.82 settles ~329 K, nominal 326 K). H7 is therefore detected
            # via the UA estimate while T is still pre-alarm; the T term only
            # requires the reactor to be on the low branch (not yet ignited).
            T_score = np.exp(-((T-332)/12)**2) if T < cstr.T_alarm else 0.2
        elif h_key == "H9_runaway_onset":
            T_score = np.exp(-((T-470)/10)**2) if T > 460 else 0.1
        else:
            T_score = 1.0

        return ua_score * ca0_score * sensor_score * T_score + 1e-9

    def update(self, obs_T, obs_Ca, enkf_state, t=None):
        """
        Bayesian update of posterior distribution.
        Returns posterior, MAP hypothesis, KL divergence from previous step.
        """
        prev_posterior = self.posterior.copy()

        # Compute likelihoods
        likelihoods = np.array([
            self.likelihood(obs_T, obs_Ca, enkf_state, h)
            for h in self.hypotheses
        ])

        # Bayes update: posterior ∝ (likelihood ** temper) × prior.
        # temper=1.0 (default) is the exact original multiplicative update.
        tl     = likelihoods if self.temper == 1.0 else likelihoods ** self.temper
        unnorm = tl * self.posterior
        if unnorm.sum() < 1e-12:
            unnorm = np.ones(self.n_hyp) / self.n_hyp
        self.posterior = unnorm / unnorm.sum()

        # KL divergence from previous posterior (belief revision signal)
        # KL(posterior || prev_posterior) — measures belief shift
        p = self.posterior + 1e-12
        q = prev_posterior + 1e-12
        kl_div = float(kl_entropy(p, q))
        self.kl_hist.append(kl_div)

        # Hypothesis revision triggered if KL exceeds threshold
        revised = False
        if self.step > 0 and kl_div > self.kl_threshold:
            revised = True
            self.revision_events.append({
                "step":self.step, "t":t, "kl":round(kl_div,4),
                "prev_map":self.hypotheses[np.argmax(prev_posterior)],
                "new_map": self.hypotheses[np.argmax(self.posterior)],
            })

        self.prior_hist.append(self.posterior.copy())
        self.step += 1

        return {
            "posterior":  self.posterior.copy(),
            "map_hyp":    self.hypotheses[np.argmax(self.posterior)],
            "map_prob":   float(np.max(self.posterior)),
            "kl_div":     round(kl_div, 4),
            "revised":    revised,
            "entropy":    round(float(kl_entropy(self.posterior+1e-12)), 4),
            "likelihoods":likelihoods.copy(),
        }

    def get_top_hypotheses(self, n=3):
        idx = np.argsort(self.posterior)[::-1][:n]
        return [(self.hypotheses[i],
                 round(float(self.posterior[i]),4),
                 FAULT_HYPOTHESES[self.hypotheses[i]]["desc"])
                for i in idx]

    def reset(self):
        self.posterior = self._initial_posterior(self.prior_spec)
        self.prior_hist = [self.posterior.copy()]
        self.kl_hist    = []
        self.revision_events = []
        self.step = 0


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 4 — CONSERVATIVE SAFETY-SCREENED CONTROLLER
# ══════════════════════════════════════════════════════════════════════════════

class ConservativeSafetyScreenedController:
    """
    Computes the protective coolant action by SAFETY SCREENING, not by information
    gain. CONSERVATIVE-UNDER-RISK state feedback (option C): it screens candidate
    setpoints on the estimated worst-case degraded plant and, when the screen verifies
    even the WEAKEST action is safe, holds the nominal action (no overcooling); the
    moment the screen cannot certify the weakest action it ESCALATES to the most
    conservative screen-verified cooling; if nothing is admissible it falls back to
    the verified-safe maximum-cooling action and the certificate fails closed
    (select()). Maximally conservative when the screen sees risk, quiet when the
    reactor is genuinely safe. The action is BELIEF-INDEPENDENT — state feedback on
    (Ca, T) via the screen; the fault posterior does not change which action is taken.

    Finding (reported, not hidden behind a label): an information-theoretic /
    expected-information-gain (EIG) action policy was investigated and found
    DEGENERATE for this safety-critical recovery problem — maximizing information
    means probing, which trades safety for information, so the EIG-greedy action is
    not admissible under the safety screen. The EIG curve is still computed and
    logged as an INVESTIGATED-AND-INACTIVE diagnostic; it does NOT drive the action
    (see NOTES.md, "P2 — conservative safety-screened controller"). This class holds
    NO decision or safety authority beyond the screen: safety is on the model-owned
    limit T_run and the gate, never on the belief or the LLM.

    (Formerly InfoTheoreticActionSelector; expected_info_gain() is retained for the
    inactive EIG diagnostic and the belief-independence figure.)
    """

    def __init__(self, cstr, n_candidates=25, horizon=5.0):
        self.cstr        = cstr
        self.n_cand      = n_candidates
        self.horizon     = horizon

    def expected_info_gain(self, Tc, Ca, T, posterior, top_hyps, temper=1.0):
        """
        Compute EIG for a candidate Tc:
        EIG = current_entropy - sum_h P(h) * entropy(posterior | obs_h)

        APPROXIMATION (documented honestly): the predictive observation under
        each hypothesis is a single DETERMINISTIC point prediction (the horizon
        endpoint T), not an expectation over a predictive observation
        distribution; the discrimination likelihood is a fixed Gaussian kernel
        (sigma=10 K). This is a tractable proxy for the true EIG.

        Performance (Phase 3.3): the per-hypothesis predicted T at this Tc does
        NOT depend on the outer hypothesis, so it is computed ONCE per candidate
        and reused, cutting simulations from ~len(top_hyps)^2 to len(top_hyps)
        with IDENTICAL numeric results (temper=1.0).
        """
        current_entropy = float(kl_entropy(posterior + 1e-12))

        # Cache predicted endpoint T per hypothesis at this Tc (independent of
        # the outer loop) — the redundant recomputation removed here.
        preds = []
        for hk, _, _ in top_hyps:
            h = FAULT_HYPOTHESES[hk]
            try:
                sol = self.cstr.simulate(
                    (0, self.horizon), [Ca, T], Tc=Tc,
                    Ca0f=np.mean(h["Ca0_range"]), UAf=np.mean(h["UA_range"]))
                preds.append(sol.y[1, -1])
            except Exception as err:
                diag_log.warning("EIG sim failed (Tc=%.1f h=%s): %s", Tc, hk, err)
                preds.append(T)

        expected_post_entropy = 0.0
        for i, (h_key, prob, _) in enumerate(top_hyps):
            T_pred   = preds[i]                  # synthetic obs if H_i were true
            sim_post = posterior.copy()
            for j in range(len(top_hyps)):
                lik_j = np.exp(-((T_pred - preds[j]) / 10) ** 2) + 1e-9
                if temper != 1.0:                # optional likelihood tempering
                    lik_j = lik_j ** temper
                sim_post[j] *= lik_j
            if sim_post.sum() > 1e-12:
                sim_post /= sim_post.sum()
            expected_post_entropy += prob * float(kl_entropy(sim_post + 1e-12))

        return current_entropy - expected_post_entropy

    def select(self, Ca, T, posterior, belief_tracker,
               safety_critical=False, UAf_screen=1.0, Ca0f_screen=1.0,
               extra_candidates=None):
        """
        CONSERVATIVE-UNDER-RISK state-feedback control (option C). 'Admissible' means
        the action's worst-case screened PEAK clears T_run (the authoritative safety
        screen, not a flag, is the arbiter — never a belief/LLM heuristic). The
        committed action is state-dependent:
          * if the WEAKEST candidate (least cooling) is screen-verified safe, there is
            NO runaway risk -> HOLD the nominal action (do not overcool a stable
            reactor); the held action is the admissible candidate nearest Tc_nom and
            is safe a fortiori (it cools more than the verified-safe weakest action);
          * the moment the screen cannot certify the weakest action, RISK is verified
            -> ESCALATE to the most conservative admissible (maximum screen-verified
            cooling);
          * if NOTHING is admissible (even maximum cooling cannot hold the screened
            plant below T_run) -> the verified-safe MAXIMUM-COOLING fallback (never
            nominal); the certificate's safe-margin check on the realized closed-loop
            peak then decides release vs fail-closed.
        Maximally conservative when the screen sees risk, quiet when the reactor is
        genuinely safe. The screen is applied at EVERY state — there is no back-off
        below T_alarm, because backing off let a degraded state REIGNITE on the
        runaway-capable model (P2); under a persisting fault the weakest action stays
        uncertifiable, so the controller stays escalated until the state is safe.

        The screen is simulated under (UAf_screen, Ca0f_screen) — the worst-case
        degraded plant from the EnKF ensemble — not the nominal plant (Phase 2.1);
        screening on the nominal plant under-predicts peak T and can pass an unsafe
        action.

        BELIEF-INDEPENDENT by construction: the action is state-feedback on (Ca, T)
        through the screen — the fault posterior never changes which action is taken,
        so the gate-3 belief-independence finding stands. The expected-information-gain (EIG)
        curve is still computed and logged as an INVESTIGATED-AND-INACTIVE diagnostic
        — it does NOT drive the action. For this safety-critical recovery problem an
        EIG-greedy policy is degenerate (probing trades safety for information); the
        belief-independence is a reported finding (NOTES.md), not a design choice
        hidden behind an 'information-theoretic' label.
        """
        top_hyps   = belief_tracker.get_top_hypotheses(3)
        # Candidates span maximum cooling (250 K, most conservative) to the weakest
        # action (345 K, least cooling). The worst-case screen gates them at every
        # state (the prior safety_critical flag is reported, not used as control).
        Tc_range   = np.linspace(250, 345, self.n_cand)
        if extra_candidates is not None and len(extra_candidates):
            # P3: merge the LLM-PROPOSED candidate setpoints into the SCREENED set,
            # clipped to the controller's authoritative actuator envelope [250, 345] K
            # (the LLM cannot propose colder than maximum cooling or weaker than the
            # weakest action). The worst-case screen still gates every candidate and
            # the conservative rule still picks the most-conservative ADMISSIBLE one,
            # so a proposal cannot change the committed action — it only enriches the
            # candidate set (connected proposal, null effect on the action).
            extra    = np.clip(np.asarray(extra_candidates, dtype=float), 250.0, 345.0)
            Tc_range = np.union1d(Tc_range, extra)
        max_cool   = float(Tc_range.min())         # 250 K — most conservative
        weakest    = float(Tc_range.max())         # 345 K — least cooling

        admissible = []      # Tc values whose worst-case screened PEAK clears T_run
        eig_curve  = []      # INACTIVE diagnostic (see docstring) — not used to select
        for Tc_cand in Tc_range:
            try:
                # Screen on the PEAK (dense n_eval), not the settled endpoint: a weak
                # action can transiently spike above T_run and settle back, and the
                # 2-point endpoint would mis-certify it as safe. T_run is a limit on
                # peak T, so the authoritative screen must use the peak (P1 numerics).
                sol_safe = self.cstr.simulate(
                    (0, self.horizon), [Ca, T], Tc=Tc_cand,
                    UAf=UAf_screen, Ca0f=Ca0f_screen, n_eval=200)
                peak_safe = float(np.max(sol_safe.y[1]))
            except Exception as err:
                diag_log.warning("safety screen sim failed (Tc=%.1f): %s",
                                 Tc_cand, err)
                peak_safe = float("inf")           # unscreenable -> inadmissible
            if peak_safe <= self.cstr.T_run:
                admissible.append(float(Tc_cand))
            # EIG is computed for the (reported) belief-independence analysis ONLY.
            eig = self.expected_info_gain(Tc_cand, Ca, T, posterior, top_hyps)
            eig_curve.append((eig, float(Tc_cand)))

        # Conservative-under-RISK selection (option C). Risk is defined ONLY by the
        # authoritative worst-case screen — never the belief or the LLM:
        #   * weakest action screen-verified safe  -> NO runaway risk -> hold the
        #     nominal action (do not overcool); use the admissible candidate nearest
        #     Tc_nom (it is screen-verified, and safe a fortiori since it cools more
        #     than the verified-safe weakest action).
        #   * weakest action NOT certifiable        -> verified risk -> ESCALATE to the
        #     most conservative admissible (maximum screen-verified cooling).
        #   * nothing admissible                     -> verified-safe MAX-COOLING
        #     fallback (never nominal); the certificate then decides fail-closed.
        # The choice is STATE-FEEDBACK on (Ca, T) via the screen, not belief-feedback:
        # the fault posterior never changes which action is taken (belief-independent).
        weakest_safe = bool(admissible) and (max(admissible) >= weakest - 1e-9)
        if not admissible:
            best_Tc, fallback_used, mode = max_cool, True, "fallback_maxcool"
        elif weakest_safe:
            best_Tc = min(admissible, key=lambda tc: abs(tc - self.cstr.Tc_nom))
            fallback_used, mode = False, "hold_nominal"
        else:
            best_Tc, fallback_used, mode = min(admissible), False, "escalate_conservative"

        # STRUCTURAL escalation (D5): the single-tank-chosen action must also clear T_run on the
        # two-CSTR-in-series shapes. If it does not but the most-conservative action (max cooling)
        # does, escalate to it -- the controller PICKS a structurally-safe action when one exists.
        # If even max cooling cannot hold the split plant, no coolant action is safe -> the gate's
        # structural-robustness check fails closed (escalation to SIS shutdown is then warranted).
        # Cheap: at most the chosen action + max cooling are screened (the gate re-verifies all).
        if HARDEN_STRUCTURAL:
            def _struct_ok(tc):
                return all(_two_cstr_peak(self.cstr, _f, [Ca, T], float(tc),
                                          Ca0f_screen, UAf_screen, 1.0, self.horizon)
                           <= self.cstr.T_run for _f in STRUCT_SPLITS)
            if not _struct_ok(best_Tc) and best_Tc > max_cool + 1e-9 and _struct_ok(max_cool):
                best_Tc, fallback_used, mode = max_cool, False, "escalate_structural"

        max_eig = max((e for e, _ in eig_curve), default=0.0)
        return {
            "Tc_selected":     round(best_Tc, 2),
            "control_mode":    mode,                 # hold_nominal | escalate_conservative | fallback_maxcool
            "fallback_used":   fallback_used,
            "n_candidates":    int(len(Tc_range)),   # screened-set size (grows with LLM proposals)
            "n_admissible":    len(admissible),
            "weakest_safe":    weakest_safe,         # worst-case screen: is doing-little safe?
            "eig":             round(max_eig, 5),    # inactive diagnostic (max over curve)
            "eig_curve":       [(round(e, 5), round(Tc, 2)) for e, Tc in eig_curve],
            "top_hyps":        top_hyps,
            "safety_critical": safety_critical,
        }


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 5 — CAUSAL ATTRIBUTION SCORER
# ══════════════════════════════════════════════════════════════════════════════

class CausalAttributionScorer:
    """
    Quantifies causal attribution of temperature deviation to each process variable.

    Implements a LOCAL SENSITIVITY analysis (Phase 5 terminology) — NOT
    Pearl do-calculus. It is a finite-difference sensitivity of the model
    output to each input, with no causal graph or interventional identification:
    Attribution(X→T) ≈ |dT/dX × ΔX| / sum_j |dT/dXj × ΔXj|

    Variables: UAf (cooling), Ca0f (feed), q_frac (flow),
               T0 (feed temperature), Tc (coolant temperature)

    FLOW SCOPE CAVEAT: q_frac is NOT estimated by the EnKF (state is
    [Ca,T,UAf,Ca0f]) and proved unobservable from (T,Ca) when augmentation was
    attempted — the estimate drifted away from truth and even invented a flow
    fault on a no-flow scenario (see FIXES.md). fingerprint() therefore leaves
    q_frac at nominal, so the flow attribution is structurally ~0 and the claim
    is scoped to cooling/feed/feed-T/control faults. Flow faults (S07/S08) are
    NOT reliably attributed.

    This gives the AXIOM agent — and the human reader — a quantified
    causal fingerprint of the fault, not just an alarm label.
    """

    def __init__(self, cstr):
        self.cstr = cstr

    def score(self, Ca, T, Tc, Ca0f, UAf, q_frac=1.0,
              T_nominal=None, rel_step=1e-2):
        """
        Compute normalized causal attribution scores for T deviation.
        Returns dict of {variable: attribution_score [0,1]}.

        Central-difference steps are a CONSISTENT relative fraction (rel_step)
        of each variable's characteristic scale (Phase 3.2), chosen well above
        the integrator tolerance so derivatives are not dominated by solver
        noise. Previously UAf/Ca0f/q_frac used a 1e-4 step (below the solver
        noise floor) while T0/Tc used +/-1 K, making the steps incomparable.
        """
        # Reference = nominal operating temperature (lower stable SS); resolve from
        # the model so it tracks T_sp rather than a hard-coded (now stale) constant.
        if T_nominal is None:
            T_nominal = self.cstr.T_sp
        T_dev = T - T_nominal
        if abs(T_dev) < 0.1:
            return {v:0.0 for v in
                    ["UAf","Ca0f","q_frac","T0","Tc"]}

        def sim_T(UAf_=UAf, Ca0f_=Ca0f, q_=q_frac,
                  T0_=None, Tc_=Tc, horizon=2.0):
            T0_ = T0_ if T0_ is not None else self.cstr.T0
            try:
                cstr_ = copy.copy(self.cstr)
                cstr_.T0 = T0_
                sol = cstr_.simulate(
                    (0,horizon),[Ca,T], Tc=Tc_, Ca0f=Ca0f_,
                    UAf=UAf_, q_frac=q_)
                return sol.y[1,-1]
            except Exception as err:
                diag_log.warning("CausalAttributionScorer.sim_T failed "
                                 "(Tc=%.1f Ca0f=%.2f UAf=%.2f): %s",
                                 Tc_, Ca0f_, UAf_, err)
                return T

        # Scale-appropriate, consistent central-difference steps.
        h_UA  = rel_step * 1.0
        h_Ca0 = rel_step * 1.0
        h_q   = rel_step * 1.0
        h_T0  = rel_step * self.cstr.T0
        h_Tc  = rel_step * self.cstr.Tc_nom
        sensitivities = {}

        # ∂T/∂UAf
        dT_dUA  = (sim_T(UAf_=UAf+h_UA) - sim_T(UAf_=UAf-h_UA)) / (2*h_UA)
        sensitivities["UAf"]    = abs(dT_dUA * (1.0 - UAf))

        # ∂T/∂Ca0f
        dT_dCa0 = (sim_T(Ca0f_=Ca0f+h_Ca0) - sim_T(Ca0f_=Ca0f-h_Ca0)) / (2*h_Ca0)
        sensitivities["Ca0f"]   = abs(dT_dCa0 * (Ca0f - self.cstr.Ca0))

        # ∂T/∂q_frac
        dT_dq   = (sim_T(q_=q_frac+h_q) - sim_T(q_=q_frac-h_q)) / (2*h_q)
        sensitivities["q_frac"] = abs(dT_dq * (1.0 - q_frac))

        # ∂T/∂T0 (feed temp disturbance; ΔT0 taken as 1 K reference)
        T0n     = self.cstr.T0
        dT_dT0  = (sim_T(T0_=T0n+h_T0) - sim_T(T0_=T0n-h_T0)) / (2*h_T0)
        sensitivities["T0"]     = abs(dT_dT0 * 1.0)

        # ∂T/∂Tc (control contribution)
        dT_dTc  = (sim_T(Tc_=Tc+h_Tc) - sim_T(Tc_=Tc-h_Tc)) / (2*h_Tc)
        sensitivities["Tc"]     = abs(dT_dTc * (self.cstr.Tc_nom - Tc))

        total = sum(sensitivities.values()) + 1e-9
        norm  = {k: round(v/total, 4) for k,v in sensitivities.items()}
        return norm

    def fingerprint(self, Ca, T, enkf_state, Tc):
        """
        Full causal fingerprint using EnKF-estimated parameters.

        EXPLANATION-ONLY (finding 2): this output flows ONLY to the LLM explanation
        prompt and to the logged trace / F5 figure. It does NOT feed the action (the
        conservative safety-screened controller ignores it — the action is computed
        before this is scored), the certificate/gate, or the diagnosis. Perturbing it
        leaves the committed action, the gate outcome, and the released decision
        unchanged; only the explanation text may differ (in live mode). Proven by
        test_attribution_is_explanation_only_not_consumed.
        """
        UAf  = enkf_state["UAf"]
        Ca0f = enkf_state["Ca0f"]
        attr = self.score(Ca, T, Tc, Ca0f, UAf)
        top  = sorted(attr.items(), key=lambda x: -x[1])
        primary_cause = top[0][0] if top[0][1] > 0.35 else "mixed"
        return {
            "attribution": attr,
            "primary_cause": primary_cause,
            "top3": top[:3],
            "fault_strength": round(abs(T - self.cstr.T_sp)/50, 3),
        }


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 5b — COUNTERFACTUAL TRAJECTORY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class CounterfactualEngine:
    """
    For each scenario, simulates what would have happened under:
    CF1: No intervention (open loop)
    CF2: Grid-search controller (exhaustive Tc grid, no causal model)
    CF3: Rule-based SIS (fixed threshold action)
    CF4: PID controller (standard feedback)

    Compares against AXIOM to quantify causal impact of each design decision.

    NOTE on the grid-search baseline: it is an exhaustive grid search over Tc
    (NOT Bayesian optimization), and its internal predictive model is given the
    TRUE Ca0f and UAf (oracle knowledge of the fault that AXIOM does not have).
    It is therefore an optimistic reference, not a like-for-like competitor.
    See FIXES.md (2.2) for the optional EnKF-based fair variant.
    """

    def __init__(self, cstr):
        self.cstr = cstr

    def no_intervention(self, t_total, y0, Ca0f, UAf, noise, dt=0.25, rng=None, uaf_fn=None):
        """Open loop — Tc held at nominal."""
        fn = lambda t,T,Ca: {"Tc": self.cstr.Tc_nom, "q_frac":1.0}
        return self.cstr.simulate_cl(t_total, y0, fn, Ca0f, UAf, noise, dt, rng=rng, uaf_fn=uaf_fn)

    def grid_search_agent(self, t_total, y0, Ca0f, UAf, noise, dt=0.5, rng=None,
                          enkf=None, uaf_fn=None):
        """
        Grid-search controller (NOT Bayesian optimization): at each step it
        scans a fixed Tc grid and picks the Tc minimizing (T_pred - T_sp)^2.

        Two variants (round 2, item 3):
        - ORACLE (enkf=None): the predictive model uses the TRUE Ca0f/UAf —
          oracle fault knowledge AXIOM does not have. Optimistic upper bound.
        - FAIR (enkf provided): the predictive model uses the SAME online EnKF
          ESTIMATES AXIOM uses (the EnKF is predicted/updated each step). This
          removes the asymmetric-information advantage: same information as
          AXIOM, only the action rule differs (grid search vs conservative safety-screened).
        """
        gen = rng if rng is not None else np.random
        last_tc = [self.cstr.Tc_nom]    # previously executed Tc (for the predict)
        def fn(t, T, Ca):
            if enkf is not None:
                enkf.predict(Tc=last_tc[0], dt=dt)   # predict under last action (item 5)
                enkf.update(np.array([T + gen.normal(0, 1.5),
                                      Ca + gen.normal(0, 0.01)]), t=t)
                es = enkf.get_state()
                ca0f_p, uaf_p = es["Ca0f"], es["UAf"]     # fair: EnKF estimates
            else:
                ca0f_p = Ca0f                              # oracle: true params
                uaf_p  = uaf_fn(t) if uaf_fn is not None else UAf  # ...incl. the current true UAf
            best_Tc, best_s = self.cstr.Tc_nom, 1e9
            for Tc_try in np.linspace(250, 350, 20):
                try:
                    sol = self.cstr.simulate((0,3),[Ca,T],Tc=Tc_try,
                                             Ca0f=ca0f_p,UAf=uaf_p)
                    s   = (sol.y[1,-1]-self.cstr.T_sp)**2
                    if s < best_s:
                        best_s, best_Tc = s, Tc_try
                except Exception as err:
                    diag_log.warning("grid_search_agent sim failed at t=%.2f "
                                     "Tc=%.1f: %s", t, Tc_try, err)
            last_tc[0] = best_Tc
            return {"Tc":best_Tc, "q_frac":1.0}
        return self.cstr.simulate_cl(t_total, y0, fn, Ca0f, UAf, noise, dt, rng=rng, uaf_fn=uaf_fn)

    def sis_agent(self, t_total, y0, Ca0f, UAf, noise, dt=0.25, rng=None, uaf_fn=None):
        """
        Rule-based SIS: fixed thresholds, no learning.
        T > T_alarm → Tc = 270K
        T > T_run   → Tc = 250K + q_frac = 0.7
        """
        cstr = self.cstr
        def fn(t, T, Ca):
            if T >= cstr.T_run:
                return {"Tc":250.0, "q_frac":0.70}
            elif T >= cstr.T_alarm:
                return {"Tc":270.0, "q_frac":1.00}
            elif T >= cstr.T_warn:
                return {"Tc":285.0, "q_frac":1.00}
            else:
                return {"Tc":cstr.Tc_nom, "q_frac":1.0}
        return self.cstr.simulate_cl(t_total, y0, fn, Ca0f, UAf, noise, dt, rng=rng, uaf_fn=uaf_fn)

    def pid_agent(self, t_total, y0, Ca0f, UAf, noise, dt=0.25,
                   Kp=8.0, Ki=0.4, Kd=1.5, rng=None, uaf_fn=None):
        """
        PID on temperature — Tc is the manipulated variable.
        Output: Tc = Tc_nom - Kp*e - Ki*∫e dt - Kd*de/dt
        """
        cstr = self.cstr
        e_prev, e_int = 0.0, 0.0
        def fn(t, T, Ca):
            nonlocal e_prev, e_int
            e      = T - cstr.T_sp
            e_int += e * dt
            de_dt  = (e - e_prev) / dt
            e_prev = e
            Tc_out = cstr.Tc_nom - Kp*e - Ki*e_int - Kd*de_dt
            Tc_out = float(np.clip(Tc_out, 250, 350))
            return {"Tc":Tc_out, "q_frac":1.0}
        return self.cstr.simulate_cl(t_total, y0, fn, Ca0f, UAf, noise, dt, rng=rng, uaf_fn=uaf_fn)


# ══════════════════════════════════════════════════════════════════════════════
# LARGE SOP KNOWLEDGE BASE (33 operating-constraint SOPs + 6 rationale REFs = 39 docs)
# Provenance: EVERY chunk traces to >=1 verified real authority. Verified citations
# live in SOURCE_REGISTRY (below); per-chunk source keys in _CORPUS_PROVENANCE (after
# the library). Full audit: CORPUS_PROVENANCE.md / corpus_provenance.json. Enforced by
# tests/test_corpus_provenance_cs1.py ("no unsourced chunk").
# ══════════════════════════════════════════════════════════════════════════════

# SOURCE_REGISTRY — the verified real authorities every corpus chunk traces to.
# kind="authority": external, independently-published, web-verified real authority
# (regulation / consensus standard / guideline / incident report / textbook /
# journal). kind="constructed": the case study's OWN design basis — the stipulated
# benchmark values, disclosed as illustrative and NEVER the sole source for a chunk.
# "verified" is the date each source's existence/title/edition was web-verified;
# internal clause/page locators flagged "[locator to confirm]" still need a copy
# check before they are printed verbatim in the manuscript.
SOURCE_REGISTRY = {
    "OSHA_PSM": {"kind":"authority","type":"regulation","verified":"2026-06-03",
        "url":"https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XVII/part-1910/subpart-H/section-1910.119",
        "citation":"OSHA, Process Safety Management of Highly Hazardous Chemicals, 29 CFR 1910.119 — incl. (d) process safety information & safe upper/lower operating limits, (f) operating procedures & emergency shutdown, (j) mechanical integrity, (m) incident investigation"},
    "CCPS_REACTIVITY": {"kind":"authority","type":"consensus-guideline","verified":"2026-06-03",
        "url":"https://www.aiche.org/ccps",
        "citation":"CCPS/AIChE, Guidelines for Chemical Reactivity Evaluation and Application to Process Design"},
    "CSB_T2": {"kind":"authority","type":"incident-report","verified":"2026-06-03",
        "url":"https://www.csb.gov/userfiles/file/t2%20final%20report.pdf",
        "citation":"U.S. Chemical Safety and Hazard Investigation Board, T2 Laboratories Inc. Runaway Reaction (Jacksonville FL, Dec 2007), Investigation Report 2008-3-I-FL"},
    "STOESSEL": {"kind":"authority","type":"textbook","verified":"2026-06-03",
        "url":"https://www.wiley.com",
        "citation":"F. Stoessel, Thermal Safety of Chemical Processes: Risk Assessment and Process Design, 2nd ed., Wiley-VCH, 2020, ISBN 978-3-527-33921-1 (MTSR, TMRad, criticality classes, adiabatic rise)"},
    "IEC_61511": {"kind":"authority","type":"consensus-standard","verified":"2026-06-03",
        "url":"https://webstore.iec.ch/en/publication/5527",
        "citation":"IEC 61511 (2016 + AMD1:2017), Functional safety — Safety instrumented systems for the process industry sector; US adoption ANSI/ISA-61511.1-2018"},
    "ISA_18_2": {"kind":"authority","type":"consensus-standard","verified":"2026-06-03",
        "url":"https://www.isa.org/products/ansi-isa-18-2-2016-management-of-alarm-systems-for",
        "citation":"ANSI/ISA-18.2-2016, Management of Alarm Systems for the Process Industries"},
    "IEC_61882": {"kind":"authority","type":"consensus-standard","verified":"2026-06-03",
        "url":"https://webstore.iec.ch/en/publication/24321",
        "citation":"IEC 61882:2016, Hazard and operability studies (HAZOP studies) — Application guide"},
    "IEC_61025": {"kind":"authority","type":"consensus-standard","verified":"2026-06-03",
        "url":"https://webstore.iec.ch/en/publication/4311",
        "citation":"IEC 61025:2006 Ed.2.0, Fault tree analysis (FTA)"},
    "VAN_HEERDEN": {"kind":"authority","type":"journal","verified":"2026-06-03",
        "url":"https://doi.org/10.1021/ie50522a030",
        "citation":"C. van Heerden, Autothermic Processes. Properties and Reactor Design, Ind. Eng. Chem. 45(6), 1242-1247 (1953)"},
    "SEMENOV": {"kind":"authority","type":"historical-theory","verified":"2026-06-03",
        "url":"https://www.wiley.com",
        "citation":"N. N. Semenov, thermal-ignition (thermal-explosion) theory, Z. Phys. 48, 571 (1928); lead citation via Stoessel (2020) for the modern English treatment"},
    "SEBORG": {"kind":"authority","type":"textbook","verified":"2026-06-03",
        "url":"https://www.wiley.com/en-us/Process+Dynamics+and+Control,+4th+Edition-p-9781119285915",
        "citation":"D. E. Seborg, T. F. Edgar, D. A. Mellichamp, F. J. Doyle III, Process Dynamics and Control, 4th ed., Wiley, 2016, ISBN 978-1-119-28591-5 (CSTR-example page [locator to confirm])"},
    "BEQUETTE": {"kind":"authority","type":"textbook","verified":"2026-06-03",
        "url":"https://www.mathworks.com/academia/books/process-dynamics-bequette.html",
        "citation":"B. W. Bequette, Process Dynamics: Modeling, Analysis, and Simulation, Prentice Hall, 1998, ISBN 0-13-206889-3 (CSTR multiplicity, pp. 506-516)"},
    "EVENSEN": {"kind":"authority","type":"journal","verified":"2026-06-03",
        "url":"https://doi.org/10.1007/s10236-003-0036-9",
        "citation":"G. Evensen, The Ensemble Kalman Filter: theoretical formulation and practical implementation, Ocean Dynamics 53, 343-367 (2003)"},
    "DESIGN_BASIS": {"kind":"constructed","type":"constructed-illustrative","verified":"n/a",
        "url":None,
        "citation":"Case-study design basis — constructed for the benchmark exothermic CSTR; model FORM anchored on Seborg/Bequette, specific VALUES stipulated by the study and disclosed as illustrative. Never the sole source for a chunk."},
}


SOP_LIBRARY = [
    {"id":"SOP-001","title":"High Temperature Response — Level 1 (Caution)",
     "text":"""Elevated temperature (T between 440-450 K) requires immediate
     attention. Increase coolant flow by 15% and reduce Tc to 290 K. Monitor
     T rate-of-rise every 30 seconds. If T rise exceeds 3 K/min, escalate to
     Level 2 response per SOP-002. Log all readings with ISO 8601 timestamps."""},

    {"id":"SOP-002","title":"High Temperature Response — Level 2 (Alarm)",
     "text":"""T exceeds 450 K: immediately set Tc to 275 K and increase
     coolant flow by 30%. Simultaneously reduce Ca0 feed by 25% to decrease
     heat generation rate. Verify UA integrity using energy balance:
     Q_removed = UA*(T-Tc). If Q_removed < 70% expected, suspect cooling
     failure and escalate to SOP-003. Notify shift supervisor immediately."""},

    {"id":"SOP-003","title":"Partial Cooling Loss Management (UA 40-65%)",
     "text":"""Diagnostic signature: T rising despite Tc setpoint maintained.
     Compute effective UA = rho_Cp*V*(dT/dt + q/V*(T-T0) + deltaH*r/rho_Cp)/(T-Tc).
     If UA_eff < 65% nominal (< 32500 cal/min/K), initiate cooling recovery:
     (1) Switch to backup cooling loop B immediately.
     (2) Reduce Ca0 feed to 0.65 mol/L to decrease heat generation by 35%.
     (3) Set Tc = 265 K on backup loop.
     (4) Verify UA recovery within 5 minutes; if not, escalate to SOP-004."""},

    {"id":"SOP-004","title":"Severe Cooling Loss Management (UA < 40%)",
     "text":"""Severe cooling failure requires emergency response.
     UA below 40% nominal (< 20000 cal/min/K) with T > 450 K is a MAJOR HAZARD.
     Actions in order: (1) Drop Tc to 250 K on all available cooling loops.
     (2) Reduce feed flow q to 70% nominal immediately to lower residence time.
     (3) Reduce Ca0 to 0.5 mol/L to halve heat generation rate.
     (4) If T continues rising beyond 460 K after 3 minutes, initiate
     controlled quench per SOP-013. Alert maintenance for heat exchanger
     inspection."""},

    {"id":"SOP-005","title":"Feed Concentration Spike Response",
     "text":"""Sudden Ca0 increase manifests as: rapid T rise with simultaneously
     elevated Ca (unreacted hot feed). Distinguish from cooling failure by:
     if Ca > 0.7 mol/L AND T rising AND UA_eff is nominal → feed spike.
     Immediate actions: (1) Verify feed control valve FCV-101 position.
     (2) Reduce Ca0 setpoint to 0.80 mol/L immediately.
     (3) Increase coolant duty by 20% preemptively.
     (4) If Ca0 surge > 1.5 mol/L, reduce feed flow q by 25% simultaneously.
     Confirm Ca0 analyzer reading against lab sample within 10 minutes."""},

    {"id":"SOP-006","title":"Feed Flow Loss / Low Flow Alarm",
     "text":"""Low feed flow (q < 70% nominal) increases reactor residence time tau,
     which can move the operating point toward the upper stable steady state.
     Diagnosis: T drifting upward with Ca decreasing (high conversion).
     Actions: (1) Check feed pump P-101 status and speed controller.
     (2) Check flow transmitter FT-101 for fouling or calibration drift.
     (3) If pump operational, check for upstream valve or strainer blockage.
     (4) Do NOT increase Tc to compensate — this masks the root cause.
     (5) Restore nominal flow rate gradually (5% increments per minute)."""},

    {"id":"SOP-007","title":"Sensor Drift — Temperature Thermocouple",
     "text":"""Thermocouple drift is identified when: measured T deviates by more
     than 10 K from energy balance estimate T_EB = T_prev + dt*(dT/dt_model).
     Cross-validate against: (1) redundant thermocouple TE-101B in same location,
     (2) infrared temperature scanner reading on reactor wall,
     (3) energy balance calculation using measured flow, Ca, and Tc.
     If deviation confirmed: (1) switch to TE-101B as primary sensor,
     (2) do NOT take aggressive corrective actions based on drifted reading,
     (3) schedule thermocouple replacement at next opportunity.
     False high T reading → risk of unnecessary cooling that suppresses
     conversion and moves reactor to lower steady state."""},

    {"id":"SOP-008","title":"Sensor Drift — Inline Ca Analyzer",
     "text":"""Inline NIR or Raman Ca analyzer drift is identified when: measured
     Ca is inconsistent with feed-minus-consumption estimate: Ca_est = Ca0 -
     r*tau. If |Ca_meas - Ca_est| > 0.15 mol/L, suspect analyzer fouling.
     Actions: (1) Collect manual sample for offline HPLC/titration.
     (2) Do not base critical decisions on Ca reading alone.
     (3) Use T as primary state variable with energy balance as backup.
     (4) Clean analyzer cell per maintenance procedure MP-CA-003."""},

    {"id":"SOP-009","title":"Incipient Cooling Degradation — Early Heat-Transfer Loss",
     "text":"""A slow upward temperature drift with the heat-transfer coefficient
     trending down (UA_eff in 75-90% of nominal) indicates INCIPIENT cooling
     degradation — early fouling or partial coolant-flow loss — caught before it
     reaches the partial/severe cooling thresholds. The reactor stays at its
     single stable steady state but at a mildly elevated temperature.
     Actions: (1) Lower Tc a few K / raise coolant flow to restore temperature margin.
     (2) Trend UA continuously per SOP-017; schedule heat-exchanger cleaning if UA
     keeps falling. (3) Tighten monitoring cadence and re-check after each step.
     (4) If UA continues below ~65% of nominal, escalate to the cooling-loss
     procedure SOP-004. Distinguish genuine heat-transfer loss from thermocouple
     drift before acting."""},

    {"id":"SOP-010","title":"Compound Fault — Cooling Loss + Feed Surge",
     "text":"""Simultaneous cooling degradation AND elevated feed creates a
     compound hazard where both heat generation increases and heat removal
     decreases. Identification: UA_eff < 70% AND Ca0 > 1.2 mol/L AND T rising.
     This is the highest-risk single-failure scenario.
     Response priority order: (1) FIRST: drop Tc to 255 K — cooling is the
     binding constraint. (2) Reduce Ca0 to 0.70 mol/L immediately.
     (3) Reduce q by 20% to lower total heat generation rate.
     (4) Declare compound fault event — two operators required.
     (5) Evaluate runaway risk index every 60 seconds using SOP-014 formula."""},

    {"id":"SOP-011","title":"Cascading Failure — Multiple Simultaneous Faults",
     "text":"""Cascading failure is declared when three or more of the following
     occur simultaneously: T > 450 K, UA_eff < 60%, Ca0 > 1.3 mol/L,
     q < 80%, AND sensor drift suspected. This requires immediate escalation.
     Actions: (1) Alert process safety officer — do not manage alone.
     (2) Begin controlled reduction of all feeds (q to 60%, Ca0 to 0.5).
     (3) Maximum cooling on all available loops (Tc = 250 K).
     (4) Prepare for emergency shutdown per SOP-015 if T > 470 K in 5 min."""},

    {"id":"SOP-012","title":"Incipient Thermal Runaway — 7 K or Less Margin",
     "text":"""When T is within 7 K of the thermal runaway threshold (T > 468 K),
     time to runaway under open-loop conditions may be less than 3 minutes.
     IMMEDIATE actions (execute in parallel, not sequence):
     (1) Set Tc = 250 K on all cooling loops NOW.
     (2) Close feed valve (q = 0) to halt heat generation.
     (3) Inject cold solvent quench (0°C solvent, 20 L/min) per SOP-013.
     (4) Activate emergency vent if vessel pressure > 90% of rated.
     (5) Page plant manager and process safety officer simultaneously."""},

    {"id":"SOP-013","title":"Emergency Cold Quench Injection Procedure",
     "text":"""Cold quench injection is authorized only when T > 468 K AND
     normal cooling has failed to arrest temperature rise.
     Quench specifications: cold water or inert solvent at T < 5°C,
     injection rate 20-30 L/min directly into reactor vessel via quench
     nozzle QN-101. Maximum quench volume: 50 L before reassessment.
     WARNING: Quench may cause rapid thermal contraction — verify vessel
     integrity after quench. Monitor for precipitation of product in cold zones.
     Post-quench: reduce Ca0 to zero, maintain cooling, analyze product."""},

    {"id":"SOP-014","title":"Runaway Risk Index Calculation",
     "text":"""The Runaway Risk Index (RRI) quantifies proximity to thermal
     runaway using the Damköhler number criterion:
     RRI = (dT/dt_rxn) / (dT/dt_cooling) = (|deltaH|*r*V) / (UA*(T-Tc) + q*rho_Cp*(T-T0)).
     RRI > 1.0 indicates heat generation exceeds removal — runaway is possible.
     RRI > 1.5 indicates imminent runaway — emergency action required.
     Compute RRI every 2 minutes when T > 440 K. Log all values.
     If RRI trend is increasing, do not wait for T_alarm — act immediately."""},

    {"id":"SOP-015","title":"Emergency Shutdown Procedure",
     "text":"""Shutdown trigger: T > 485 K, OR RRI > 1.8 for > 2 min,
     OR two unresolved alarms for > 5 minutes.
     Shutdown sequence (strict order):
     (1) t=0s: Close main feed valve FV-101 (q = 0).
     (2) t=5s: Open emergency cooling bypass (Tc → 250 K all loops).
     (3) t=10s: Activate emergency vent EV-101 if P > 90% rated.
     (4) t=30s: Initiate cold quench per SOP-013 if T still rising.
     (5) t=60s: Notify process safety officer and plant manager.
     (6) After T < 400 K: begin controlled depressurization.
     Do not restart without written authorization from engineering team."""},

    {"id":"SOP-016","title":"Post-Incident Root Cause Analysis Protocol",
     "text":"""Following any T > 460 K event, a formal root cause analysis (RCA)
     is required within 48 hours. RCA must address:
     (1) Causal chain: which parameter deviated first and why?
     (2) Detection lag: how long between fault onset and alarm?
     (3) Response time: time from alarm to corrective action initiation.
     (4) Barrier effectiveness: did existing controls perform as designed?
     (5) Counterfactual: what is the estimated outcome if no action was taken?
     Document using AXIOM decision log and sensor historian data.
     Apply 5-Why methodology and fault tree analysis per IEC 61025."""},

    {"id":"SOP-017","title":"Heat Transfer Coefficient Monitoring and Trending",
     "text":"""UA should be trended continuously using online heat balance:
     UA_calc = (rho_Cp*V*dT/dt + q*rho_Cp*(T-T0) + |deltaH|*r*V) / (T-Tc).
     Alert thresholds: UA < 90% nominal → maintenance notification.
     UA < 75% → schedule cleaning of heat exchanger HX-101.
     UA < 60% → activate backup cooling loop immediately.
     UA < 40% → declare cooling emergency per SOP-004.
     Typical UA degradation causes: fouling (gradual), tube rupture (sudden),
     coolant pump cavitation (fluctuating UA signal)."""},

    {"id":"SOP-018","title":"Reactant Feed Quality Monitoring",
     "text":"""Feed concentration Ca0 must be verified against:
     (1) Upstream tank level and inventory balance.
     (2) Inline density or refractive index measurement.
     (3) Feed forward Ca0 analyzer AZ-101.
     If Ca0 > 1.15 mol/L for > 5 minutes, reduce feed concentration
     immediately by diluting with solvent stream SV-201.
     Maximum allowed Ca0: 1.5 mol/L (above this, heat generation exceeds
     design basis at nominal cooling). If Ca0 spike is unexplained,
     check upstream batch tank TK-101 for mixing failure."""},

    {"id":"SOP-019","title":"Safe Operating Region Boundary Management",
     "text":"""At nominal parameters the exothermic CSTR has a SINGLE stable steady
     state — the adiabatic temperature rise is too small for ignition/extinction
     multiplicity. The hazard is not a jump between states but FAULT-INDUCED
     runaway: loss of cooling (UA decay) or a feed-concentration surge raises heat
     generation relative to removal and drives the single state to high temperature.
     Safe operating region: T in [320, 435] K (the nominal steady state ~326 K up to
     comfortably below the warning threshold T_warn = 440 K); reactant Ca consistent with
     the single stable steady state (~0.98 mol/L at the cool nominal point, falling toward
     ~0.10 as conversion rises under elevated temperature).
     If the operating point exits the safe region, aggressive corrective action is
     needed (restore cooling, cut feed concentration/flow, drop Tc). Monitor the
     runaway risk index per SOP-014. There is no fold or bifurcation locus to
     track at these parameters."""},

    {"id":"SOP-020","title":"Cooling Jacket Integrity Verification",
     "text":"""Verify cooling jacket integrity quarterly and after any
     temperature excursion above 460 K.
     Tests: (1) Pressure test jacket at 1.5x operating pressure.
     (2) Check inlet/outlet coolant temperatures for bypass (if Tc,out ≈ Tc,in,
     suspect bypass or pump failure).
     (3) Verify coolant flow rate against design FT-201.
     (4) Inspect for corrosion on external jacket surface.
     After any repair: perform UA verification run at 3 operating points
     before resuming normal operation."""},

    {"id":"SOP-021","title":"Control System Failure — DCS Fallback",
     "text":"""If Distributed Control System (DCS) fails during operation:
     (1) Switch all controllers to manual mode.
     (2) Set Tc manually to 295 K (conservative safe value).
     (3) Set q manually to 85% nominal.
     (4) Assign dedicated operator to monitor T and Ca manually every 2 min.
     (5) Do not attempt DCS restart while reactor is above 430 K.
     Maximum allowed manual operation duration: 60 minutes.
     If DCS not restored in 60 min, initiate controlled shutdown per SOP-015."""},

    {"id":"SOP-022","title":"Operator Shift Handover — Safety Critical Information",
     "text":"""All shift handovers must include: (1) Current T, Ca, UA_eff, Ca0.
     (2) Active alarms and their duration.
     (3) Any SOP deviations taken in past shift with justification.
     (4) Maintenance activities planned or in progress.
     (5) Trend of past 4 hours of T and UA_eff.
     (6) Status of all backup systems (cooling loop B, quench system, DCS).
     Handover checklist must be signed by both outgoing and incoming operators.
     Do not accept handover if T > 440 K without senior supervisor present."""},

    {"id":"SOP-023","title":"Arrhenius Parameter Verification and Recalibration",
     "text":"""Kinetic parameters (k0, Ea/R) must be recalibrated annually
     or after any major feedstock quality change.
     Calibration procedure: (1) Run reactor at 5 steady-state conditions
     spanning T = 330-430 K (bracketing the ~326 K nominal steady state up toward the
     alarm band). (2) Measure r = (Ca0 - Ca) / tau at each SS.
     (3) Fit k0 and Ea/R via linear regression on ln(r/Ca) vs 1/T.
     (4) Compare to previous values — if Ea/R changes > 5%, suspect
     catalyst deactivation or impurity in feed.
     (5) Update all process models and DCS parameter tables.
     Any change in k0 > 20% requires re-evaluation of all SOP temperature limits."""},

    {"id":"SOP-024","title":"Bayesian Alarm Rationalization Procedure",
     "text":"""When multiple alarms are active simultaneously, operators must
     prioritize using the following Bayesian framework:
     (1) Identify the most likely root cause given current process state.
     (2) Address the root cause, not the symptoms.
     (3) If root cause is uncertain, take the conservative action for the
     most hazardous plausible hypothesis.
     (4) Design a diagnostic action that distinguishes between competing
     hypotheses before committing to irreversible interventions.
     Principle: one root cause typically explains all active alarms.
     Alarm flood management: silence nuisance alarms only after root cause
     identified; never silence alarms before understanding their cause."""},

    {"id":"SOP-025","title":"Agentic AI Safety Override and Human-in-the-Loop",
     "text":"""AXIOM agentic AI recommendations require human authorization
     before execution for: (1) Any Tc change exceeding 30 K from setpoint.
     (2) Any feed flow reduction exceeding 20%.
     (3) Emergency quench initiation.
     (4) Emergency shutdown initiation.
     Automatic execution permitted for: Tc adjustments within ±20 K of setpoint,
     feed concentration reduction < 25%, alarm acknowledgment and logging.
     Human override: operator can reject any AXIOM recommendation by pressing
     OVERRIDE button — this logs the override and AXIOM adapts its next
     recommendation. All AXIOM actions logged with timestamp and rationale."""},

    # ── Operating-constraint tier additions (KB Phase B) — fill the audited gaps.
    {"id":"SOP-026","title":"Operating Envelope and Design Basis Summary",
     "tier":"operating-constraint",
     "text":"""Authoritative operating envelope and design basis. Nominal operating point:
     the lower stable steady state at T_sp = 326.2 K with feed Ca0 = 1.0 mol/L, feed flow
     q = 100 L/min and volume V = 100 L (residence time tau = V/q = 1.0 min), feed
     temperature T0 = 350 K, nominal coolant Tc = 300 K. Coolant actuator range: Tc is
     adjustable from 250 K (maximum cooling) to 345 K (minimum cooling). Design heat
     transfer UA = 5.0e4 cal/min/K. Temperature thresholds: warning 440 K, alarm 450 K,
     RUNAWAY LIMIT 475 K (the controlling hard safety limit), emergency shutdown 485 K.
     Composition: nominal Ca0 1.0 mol/L; maximum admissible Ca0 1.5 mol/L (above this, heat
     generation exceeds the design cooling at nominal Tc). Every protective action must keep
     the reactor temperature below the 475 K runaway limit with margin. This is the
     limit-bearing design-basis document; the protection system ENFORCES that limit from the
     validated process model, not by reading this text, so a stale or edited copy of this
     document cannot move the enforced limit."""},

    {"id":"SOP-027","title":"Coolant Circulation / Flow Reduction Response",
     "tier":"operating-constraint",
     "text":"""Coolant-side flow reduction (jacket circulation loss: coolant pump P-201 trip,
     valve closure, fouling, or cavitation) cuts effective heat removal even with Tc
     commanded correctly. Signature: T rising while the Tc setpoint is held, with the coolant
     temperature rise (Tc,out - Tc,in) narrowing or jacket flow FT-201 reading low.
     Distinguish from UA fouling (gradual, SOP-009/017) by the coolant-flow / delta-T signal.
     Actions: (1) Confirm coolant pump P-201 status and jacket flow FT-201. (2) Switch to
     backup coolant loop B. (3) Command maximum cooling (Tc = 250 K) on the available loop
     while flow is restored. (4) If flow cannot be restored and T approaches the alarm
     threshold, cut feed concentration and flow to reduce heat generation per SOP-004.
     (5) Raising the commanded Tc does NOT help a flow loss - treat it as a coolant-delivery
     fault, not a setpoint problem."""},

    {"id":"SOP-028","title":"Feed Temperature Excursion Response",
     "tier":"operating-constraint",
     "text":"""An elevated feed temperature T0 (preheater fault, loss of feed cooling) raises
     the reactor heat load independently of feed concentration. Nominal T0 = 350 K.
     Signature: T drifting up with Ca0 and UA nominal and feed-temperature TI-102 reading
     high. Actions: (1) Verify feed temperature TI-102 against the preheater control.
     (2) Increase coolant duty (lower Tc toward maximum cooling) to absorb the added sensible
     heat. (3) Restore T0 to 350 K via the preheater bypass. (4) If T0 cannot be restored and
     reactor T approaches the warning threshold, reduce feed concentration to lower total heat
     generation. Distinguish from a concentration spike (SOP-005) by the analyzer: a
     feed-temperature excursion shows normal Ca0."""},

    {"id":"SOP-029","title":"Alarm and Interlock Rationalization / SIS Trip Setpoints",
     "tier":"operating-constraint",
     "text":"""Safety-instrumented thresholds, consistent with the design basis (SOP-026).
     Warning 440 K (operator attention; increase cooling). Alarm 450 K (high priority;
     execute SOP-002). Runaway limit 475 K (the controlling hard limit; protective action
     must keep T below it with margin). SIS high-temperature trip 485 K (independent
     safety-instrumented function: close feed FV-101, command maximum cooling, open the
     emergency vent on over-pressure). Rate-of-rise alarm: dT/dt > 3 K/min. The SIS trip is
     independent of the basic control layer and of any AI recommendation and is keyed to the
     model-owned thresholds. Alarm-flood handling follows ISA-18.2 rationalization: one root
     cause typically explains the flood (see SOP-024)."""},

    {"id":"SOP-030","title":"HAZOP / Safety-Case Summary - Exothermic CSTR Runaway",
     "tier":"operating-constraint",
     "text":"""Representative HAZOP / safety-case summary. Hazard: thermal runaway driven by
     the positive feedback between Arrhenius kinetics and reactor temperature when heat
     generation exceeds removal. Causes (deviations): loss of cooling (UA degradation -
     fouling, tube fault, coolant-flow loss), feed-concentration surge, feed-temperature rise,
     or feed-flow reduction (raising residence time). Consequence: temperature excursion
     toward the runaway limit (475 K) and, unmitigated, past the shutdown trip (485 K); the
     adiabatic temperature rise (~150 K) is sufficient to over-pressure the vessel. Safeguards
     (independent layers): basic control (coolant setpoint, feed control); high-temperature
     alarm (450 K); the conservative safety-screened protective controller; the SIS
     high-temperature trip (485 K); emergency cold quench (SOP-013); and emergency relief.
     Controllability boundary: a runaway initiated from a high in-situ reactant charge at
     elevated temperature can outrun the cooling system, so the protective strategy must
     commit aggressive cooling before that point. Basis: van Heerden (1953) multiplicity
     analysis and Semenov ignition theory (rationale tier)."""},

    {"id":"SOP-031","title":"Equipment Datasheet - Reactor and Cooling System",
     "tier":"operating-constraint",
     "text":"""Reactor vessel R-101: jacketed CSTR, working volume V = 100 L, exothermic
     liquid-phase reaction. Feed: nominal flow q = 100 L/min (residence time tau = 1.0 min at
     nominal), nominal concentration Ca0 = 1.0 mol/L, feed temperature T0 = 350 K. Cooling:
     jacket with design heat transfer UA = 5.0e4 cal/min/K; coolant supply adjustable over
     250-345 K (nominal 300 K); backup coolant loop B; quench nozzle QN-101. Reaction medium:
     density x heat capacity rho_Cp = 500 cal/L/K. Characterized kinetics (first order in
     reactant): k0 = 7.2e9 /min, Ea/R = 8750 K, heat of reaction dH = -75000 cal/mol
     (adiabatic rise ~150 K). Instrumentation: reactor thermocouples TE-101A/B, inline Ca
     analyzer AZ-101, feed flow FT-101, jacket flow FT-201, feed temperature TI-102."""},

    {"id":"SOP-032","title":"Control Narrative - Conservative Protective Strategy",
     "tier":"operating-constraint",
     "text":"""Control narrative. The coolant setpoint Tc is the primary manipulated variable
     for reactor temperature; feed concentration and flow are secondary handles. Under fault
     or uncertainty the protective strategy is CONSERVATIVE and safety-screened: it holds the
     least-aggressive coolant action only when the worst-case screen verifies that action
     keeps the reactor below the runaway limit (475 K) with margin; otherwise it escalates
     toward more aggressive cooling, with maximum cooling (Tc = 250 K) as the verified-safe
     fallback. The action is chosen by the screen against the model-owned limit, never by the
     diagnosis or the narrative; a confident diagnosis improves explanation and efficiency but
     never widens the safety envelope. Operators retain override authority (SOP-025). This
     narrative documents intent; the enforced limits live in the process model."""},

    {"id":"SOP-033","title":"Incident Write-up - Runaway Near-Miss and Validated Recovery",
     "tier":"operating-constraint",
     "text":"""Representative validated incident (constructed for this case study, not a real
     proprietary record). Event: a partial cooling-loss fault (UA degraded to ~50% by jacket
     fouling) developed over several minutes while feed load was elevated; reactor temperature
     entered the alarm band (~452 K) with the in-situ reactant largely consumed (Ca ~0.10).
     Response: the protective controller's worst-case screen rejected the under-cooling
     candidates and committed aggressive cooling (Tc lowered toward 250 K) with feed
     reduction; the closed-loop peak was held near 461 K, below the 475 K runaway limit, and
     the reactor recovered to the nominal steady state. Lesson: at low in-situ reactant the
     runaway is controllable IF aggressive cooling is committed early, whereas under-cooling
     lets the elevated feed re-drive the excursion. This validated recovery is the basis for
     the conservative escalate-under-risk rule (SOP-032); a higher in-situ reactant charge at
     the same temperature would have been past the controllable point (SOP-030)."""},

    # ── Rationale / literature tier (KB Phase B) — REAL, established works grounding the
    # engineering rationale. NON-authoritative: never a source of hard limits or feasibility
    # (excluded from the default constraint retrieval channel). Any locator not verifiable
    # offline is marked [Reference needed] rather than invented.
    {"id":"REF-001","title":"van Heerden (1953) - autothermic processes / reactor multiplicity",
     "tier":"rationale",
     "text":"""C. van Heerden, "Autothermic Processes: Properties and Reactor Design,"
     Industrial and Engineering Chemistry, 45(6), 1242-1247 (1953). DOI 10.1021/ie50522a030.
     Classical analysis of steady-state multiplicity and thermal stability in continuous
     exothermic reactors: the heat-generation vs heat-removal balance and the stability of
     operating points. Grounds the runaway-hazard rationale; not a source of plant limits."""},

    {"id":"REF-002","title":"Semenov thermal-ignition theory (via Stoessel; 1928 primary)",
     "tier":"rationale",
     "text":"""Semenov thermal-explosion (ignition) theory: the critical balance between heat
     release and heat dissipation governing thermal ignition; foundational to runaway criteria.
     Working citation: presented in modern, citable English form by Stoessel (REF-005); lead
     with that for any claim. Historical primary (optional): N. N. Semenov, Zeitschrift fur
     Physik 48, 571 (1928) [Z. Phys. 48, 571 -- not Z. Phys. Chem.]. Rationale only; not a
     source of limits."""},

    {"id":"REF-003","title":"Seborg, Edgar, Mellichamp & Doyle - Process Dynamics and Control (4th ed., 2016)",
     "tier":"rationale",
     "text":"""D. E. Seborg, T. F. Edgar, D. A. Mellichamp, F. J. Doyle III, "Process Dynamics
     and Control," 4th ed., Wiley, 2016. ISBN 978-1-119-28591-5. Standard reference for the CSTR
     model form, dynamics, and control; the kinetic / heat-balance structure used in this case
     study is anchored on this text. (Exact CSTR-example page to be confirmed against the
     physical 4th-ed copy -- a page-locator confirmation, not an unverified reference.) Rationale
     only; not a source of limits."""},

    {"id":"REF-004","title":"Bequette - Process Dynamics: Modeling, Analysis, and Simulation (1998)",
     "tier":"rationale",
     "text":"""B. W. Bequette, "Process Dynamics: Modeling, Analysis, and Simulation," Prentice
     Hall, 1998. ISBN 0-13-206889-3. CSTR modeling, steady-state multiplicity, and dynamic
     simulation (CSTR material approximately pp. 506-516). NOTE: this is the 1998 "Process
     Dynamics" title, NOT the later 2003 "Process Control: Modeling, Design and Simulation."
     Rationale only; not a source of limits."""},

    {"id":"REF-005","title":"Stoessel - Thermal Safety of Chemical Processes (2nd ed., 2020)",
     "tier":"rationale",
     "text":"""F. Stoessel, "Thermal Safety of Chemical Processes: Risk Assessment and Process
     Design," 2nd ed., Wiley-VCH, 2020. ISBN 978-3-527-33921-1. Adiabatic temperature rise,
     time-to-maximum-rate, and runaway assessment; the basis for the ~150 K adiabatic-rise
     framing and the working citation for Semenov ignition theory (REF-002). (Edition
     author-confirmed: 2nd ed., 2020; any specific chapter/page locator, if later cited, to be
     confirmed against that copy.) Rationale only; not a source of limits."""},

    {"id":"REF-006","title":"Evensen - The Ensemble Kalman Filter (2003)",
     "tier":"rationale",
     "text":"""G. Evensen, "The Ensemble Kalman Filter: theoretical formulation and practical
     implementation," Ocean Dynamics 53, 343-367 (2003). DOI 10.1007/s10236-003-0036-9. (See
     also G. Evensen, "Data Assimilation: The Ensemble Kalman Filter," Springer, 2009.) Basis for
     the EnKF state/parameter estimation (reactor state and UA/Ca0). Rationale only; not a source
     of limits."""},

    # ── Corpus expansion (2026-06): additional operating-constraint SOPs and rationale REFs,
    # authored in the SAME illustrative-anchored-to-real-authority convention as SOP-001..033
    # (see _CORPUS_PROVENANCE). No NEW 400-500 K limit values are introduced (the documented
    # set 440/450/460/475/485 K is reused), so the knowledge-fidelity ground truth is unchanged.
    {"id":"SOP-034","title":"Time-to-Maximum-Rate (TMRad) Criticality Screen",
     "text":"""On a confirmed loss of cooling, estimate the time to maximum rate under
     adiabatic conditions (TMRad). If TMRad is under 8 hours at the current temperature,
     treat the event as high criticality: initiate maximum cooling and feed cutback
     immediately per SOP-003 and prepare for emergency quench (SOP-038). Document the
     TMRad estimate and the assumed reaction enthalpy.""",
     "keywords":["TMRad","time to maximum rate","cooling loss","criticality","adiabatic","runaway"]},

    {"id":"SOP-035","title":"Reaction Criticality Class Determination",
     "text":"""Classify the scenario by the ordering of the maximum temperature of the
     synthesis reaction (MTSR) relative to the boiling/technical limit and the runaway
     onset: criticality classes 1-2 are controllable by process means, classes 3-5 require
     independent protection layers and an armed safety instrumented function. Re-classify
     whenever feed concentration or cooling capacity changes.""",
     "keywords":["criticality class","MTSR","independent protection","runaway onset","reactivity"]},

    {"id":"SOP-036","title":"Safety Instrumented Function Proof Test",
     "text":"""The high-temperature safety instrumented function (SIF) that trips the reactor
     at the 485 K shutdown setpoint shall be proof-tested at the interval required to meet its
     target safety integrity level. Verify the sensor, logic solver, and final element trip on
     demand; record the as-found/as-left state and any failures for SIL verification.""",
     "keywords":["SIS","SIF","proof test","SIL","shutdown","485 K","interlock","trip"]},

    {"id":"SOP-037","title":"Layer-of-Protection / Independent Protection Layer Audit",
     "text":"""For the cooling-loss runaway scenario, verify each credited independent protection
     layer (basic control, alarm with operator response, the safety instrumented function, and
     emergency relief) is genuinely independent, auditable, and adequate. A layer shared with the
     initiating cause may not be credited. Record the protection-layer accounting.""",
     "keywords":["LOPA","independent protection layer","IPL","runaway","relief","compound"]},

    {"id":"SOP-038","title":"Emergency Cooling and Quench Activation",
     "text":"""On a confirmed runaway trend approaching the 475 K limit, activate emergency
     cooling: drive the coolant setpoint to the 250 K actuator floor (maximum cooling) and, if
     the trend continues, initiate the quench/dump system. This is a safe-state action; it takes
     precedence over production targets. Confirm via the trusted model that peak temperature is
     contained below 475 K.""",
     "keywords":["emergency cooling","quench","maximum cooling","runaway","475 K","coolant"]},

    {"id":"SOP-039","title":"Loss of Cooling Utility (Site Utility Failure)",
     "text":"""On loss of the cooling-water or chilled-coolant utility, treat heat-removal
     capacity as degraded regardless of the local controller demand: cut the reactant feed,
     transition to the safe holding state, and escalate per SOP-003. Do not rely on the jacket
     until utility supply and flow are confirmed restored.""",
     "keywords":["loss of cooling utility","utility failure","feed cutback","heat removal","cooling"]},

    {"id":"SOP-040","title":"Reaction Thermal-Screening Requirement",
     "text":"""Before a campaign, the reaction mixture shall have calorimetric thermal screening
     (e.g. DSC followed by adiabatic calorimetry) on file, documenting the exotherm onset and the
     adiabatic temperature rise (about 150 K for this system). Operating envelopes and protection
     layers shall be consistent with the screened thermal data.""",
     "keywords":["thermal screening","DSC","calorimetry","adiabatic rise","exotherm","reactivity"]},

    {"id":"SOP-041","title":"High-Temperature Alarm Rationalization",
     "text":"""Maintain a rationalized alarm hierarchy for the reactor temperature: the 440 K
     caution and 450 K alarm thresholds each carry a defined operator response and adequate
     response time. During an upset, suppress nuisance and flood alarms per the alarm-management
     program so the operator is not overwhelmed; never suppress the high-temperature safety alarm.""",
     "keywords":["alarm rationalization","alarm flood","440 K","450 K","operator response","alarm management"]},

    # Versioned stale-trap (C1, symmetric with the column's SPEC-DIST-PURITY v2.0/v1.0): the SIS
    # trip setpoint was REVISED downward; the CURRENT value is gated, the SUPERSEDED value is a
    # planted distractor so version-grounding is MEASURABLE (a proposer that states the old trip is
    # detected as STALE, exactly like CS2's stale purity spec). No NEW current limit is introduced
    # (485 K is already documented); only the superseded 495 K is added, carried by a chunk marked
    # superseded so it is excluded from the documented-limit ground truth.
    {"id":"SOP-042","title":"Emergency Shutdown (SIS) Trip Setpoint — current",
     "source_id":"SIS-TRIP", "version":"2.0",
     "keywords":["SIS","shutdown","trip","setpoint","interlock","485"],
     "text":"""CURRENT (rev 2.0): the safety instrumented system trips the reactor to
     emergency shutdown at 485 K. This setpoint supersedes the previous revision; use only
     this value. The hard runaway limit remains enforced by the process model."""},
    {"id":"SOP-043","title":"Emergency Shutdown (SIS) Trip Setpoint — SUPERSEDED",
     "source_id":"SIS-TRIP", "version":"1.0", "superseded":True,
     "keywords":["SIS","shutdown","trip","setpoint","stale","superseded","495"],
     "text":"""OUTDATED (rev 1.0): SIS trip at 495 K. Retained for audit only; do NOT use --
     superseded by rev 2.0 (485 K). Citing this 495 K value is a stale-knowledge error."""},

    {"id":"REF-007","title":"Stoessel - cooling-failure runaway and the criticality framework",
     "tier":"rationale",
     "text":"""F. Stoessel, "Thermal Safety of Chemical Processes," 2nd ed., Wiley-VCH, 2020.
     The cooling-failure scenario and the MTSR / TMRad / criticality-class framework that underpin
     SOP-034/-035/-040: severity from the adiabatic rise, probability from the time to maximum
     rate. Rationale only; not a source of limits."""},

    {"id":"REF-008","title":"IEC 61511 / CCPS - independent protection and SIS rationale",
     "tier":"rationale",
     "text":"""IEC 61511 (2016) and CCPS layer-of-protection guidance: the basis for treating the
     safety instrumented function and the other protection layers as independent, proof-tested,
     and SIL-verified (SOP-036/-037). Rationale only; not a source of limits."""},
]


# _CORPUS_PROVENANCE — per-chunk source keys (into SOURCE_REGISTRY), how the chunk
# was derived, and its cleanliness grade (A cleanly sourced / B sourced principle +
# illustrative numbers / C constructed design-basis|method, disclosed). Every entry
# carries >=1 authority key; DESIGN_BASIS only ever appears ALONGSIDE an authority.
# See CORPUS_PROVENANCE.md for the full table and the manuscript disclosure paragraph.
_CORPUS_PROVENANCE = {
    "SOP-001": {"sources":["ISA_18_2","OSHA_PSM"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-002": {"sources":["ISA_18_2","OSHA_PSM","CCPS_REACTIVITY"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-003": {"sources":["CCPS_REACTIVITY","CSB_T2","STOESSEL"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-004": {"sources":["CCPS_REACTIVITY","CSB_T2","STOESSEL","OSHA_PSM"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-005": {"sources":["CCPS_REACTIVITY","OSHA_PSM"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-006": {"sources":["SEBORG","BEQUETTE","CCPS_REACTIVITY"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-007": {"sources":["OSHA_PSM","SEBORG"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-008": {"sources":["OSHA_PSM","SEBORG"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-009": {"sources":["CCPS_REACTIVITY","STOESSEL"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-010": {"sources":["IEC_61882","CCPS_REACTIVITY"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-011": {"sources":["IEC_61882","CCPS_REACTIVITY","OSHA_PSM"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-012": {"sources":["STOESSEL","CCPS_REACTIVITY"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-013": {"sources":["CCPS_REACTIVITY","STOESSEL","OSHA_PSM"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-014": {"sources":["CCPS_REACTIVITY","STOESSEL","SEMENOV","VAN_HEERDEN"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-015": {"sources":["OSHA_PSM","IEC_61511"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-016": {"sources":["IEC_61025","CCPS_REACTIVITY","CSB_T2","OSHA_PSM"],"derivation":"paraphrased","cleanliness":"A"},
    "SOP-017": {"sources":["CCPS_REACTIVITY","SEBORG"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-018": {"sources":["CCPS_REACTIVITY","OSHA_PSM"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-019": {"sources":["VAN_HEERDEN","SEBORG","BEQUETTE"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-020": {"sources":["OSHA_PSM"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-021": {"sources":["IEC_61511"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-022": {"sources":["OSHA_PSM"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-023": {"sources":["SEBORG","BEQUETTE"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-024": {"sources":["ISA_18_2"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-025": {"sources":["IEC_61511","DESIGN_BASIS"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "SOP-026": {"sources":["OSHA_PSM","SEBORG","BEQUETTE","DESIGN_BASIS"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "SOP-027": {"sources":["CCPS_REACTIVITY","CSB_T2"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-028": {"sources":["CCPS_REACTIVITY","SEBORG"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-029": {"sources":["IEC_61511","ISA_18_2"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-030": {"sources":["IEC_61882","VAN_HEERDEN","SEMENOV","CCPS_REACTIVITY"],"derivation":"paraphrased","cleanliness":"A"},
    "SOP-031": {"sources":["OSHA_PSM","SEBORG","BEQUETTE","DESIGN_BASIS"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "SOP-032": {"sources":["IEC_61511","DESIGN_BASIS"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "SOP-033": {"sources":["OSHA_PSM","CSB_T2","DESIGN_BASIS"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "REF-001": {"sources":["VAN_HEERDEN"],"derivation":"verbatim","cleanliness":"A"},
    "REF-002": {"sources":["SEMENOV","STOESSEL"],"derivation":"verbatim","cleanliness":"A"},
    "REF-003": {"sources":["SEBORG"],"derivation":"verbatim","cleanliness":"A"},
    "REF-004": {"sources":["BEQUETTE"],"derivation":"verbatim","cleanliness":"A"},
    "REF-005": {"sources":["STOESSEL"],"derivation":"verbatim","cleanliness":"A"},
    "REF-006": {"sources":["EVENSEN"],"derivation":"verbatim","cleanliness":"A"},
    # corpus expansion (2026-06) -- each anchored to >=1 verified real authority
    "SOP-034": {"sources":["STOESSEL","CCPS_REACTIVITY"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-035": {"sources":["STOESSEL","CCPS_REACTIVITY"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-036": {"sources":["IEC_61511"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-037": {"sources":["CCPS_REACTIVITY","IEC_61511","IEC_61882"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-038": {"sources":["CCPS_REACTIVITY","STOESSEL","OSHA_PSM"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-039": {"sources":["CCPS_REACTIVITY","OSHA_PSM"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-040": {"sources":["CCPS_REACTIVITY","STOESSEL"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-041": {"sources":["ISA_18_2"],"derivation":"synthesized-from-source","cleanliness":"B"},
    "SOP-042": {"sources":["IEC_61511","OSHA_PSM"],"derivation":"constructed-illustrative","cleanliness":"B"},
    "SOP-043": {"sources":["IEC_61511","OSHA_PSM"],"derivation":"constructed-illustrative","cleanliness":"B"},
    "REF-007": {"sources":["STOESSEL"],"derivation":"paraphrased","cleanliness":"B"},
    "REF-008": {"sources":["IEC_61511","CCPS_REACTIVITY"],"derivation":"paraphrased","cleanliness":"B"},
}

# 0C.10 -- flag-gated retrieval DISTRACTORS (OFF by default -> the locked 51-chunk corpus and its
# retrieval_eval are byte-for-byte unchanged). These are confusable-but-IRRELEVANT chunks on adjacent
# admin/maintenance topics: a keyword retriever is tempted to surface them (they mention temperature/
# coolant/feed), but they are NEVER in any FAMILY_SOPS relevant set, so a better (dense) retriever
# ranks them below the real fault-response SOPs -> lets dense-vs-lexical separate (figure C6). They are
# provenance-tracked like every chunk. Enabled for the re-run with AXIOM_DISTRACTORS=1 (set before import).
CS1_DISTRACTORS = [
    {"id":"DIST-001","title":"Thermocouple Calibration Schedule (routine)",
     "text":"Temperature thermocouples and coolant-loop RTDs are calibrated on a 12-month interval "
            "per the instrument management program; record drift and calibration dates. Routine "
            "scheduling only -- this is NOT a fault response or operating action."},
    {"id":"DIST-002","title":"Coolant Supply Procurement Specification",
     "text":"Procurement of cooling-water and glycol coolant: approved suppliers, purity grade, "
            "delivery cadence, and minimum on-site inventory. Administrative procurement only -- NOT "
            "an operating or fault-response procedure."},
    {"id":"DIST-003","title":"Feed Tank External Coating and Corrosion Inspection",
     "text":"Feed storage tanks: external coating condition and corrosion monitoring on the "
            "mechanical-integrity schedule; plan recoating. Maintenance scheduling only -- unrelated "
            "to any feed-composition or feed-flow process upset."},
    {"id":"DIST-004","title":"Operator Training and Competency Records",
     "text":"Operator training matrix and competency re-certification cadence for the reactor unit, "
            "including refresher modules on temperature alarms and cooling procedures. Administrative "
            "training records only -- NOT an operating action or fault response."},
    {"id":"DIST-005","title":"Coolant Pump Lubrication and PM Schedule",
     "text":"Preventive-maintenance lubrication intervals for the coolant circulation pump bearings and "
            "seals; grease specification and work-order cadence. Routine mechanical maintenance only -- "
            "NOT a cooling-loss fault response."},
    {"id":"DIST-006","title":"Management of Change (MOC) Paperwork Routing",
     "text":"Administrative routing and approval signatures for a management-of-change request that "
            "touches reactor temperature setpoints; form numbers and review queue. Document-control "
            "workflow only -- NOT an operating procedure or safety action."},
    {"id":"DIST-007","title":"Personal Protective Equipment Inventory",
     "text":"PPE stock levels and reorder points for the reactor area: heat-resistant gloves, face "
            "shields, and chemical aprons. Procurement/inventory administration only -- unrelated to "
            "any process temperature, cooling, or feed upset."},
    {"id":"DIST-008","title":"Calibration Gas Cylinder Storage",
     "text":"Storage, segregation, and expiry tracking of calibration gas cylinders used for the gas "
            "detectors near the reactor. Stores administration only -- NOT a sensor-fault response or "
            "a temperature-measurement action."},
    {"id":"DIST-009","title":"Annual Insurance Loss-Prevention Survey Logistics",
     "text":"Scheduling and document-gathering logistics for the annual insurer loss-prevention walkdown "
            "of the reactor building. Administrative coordination only -- NOT a hazard analysis, operating "
            "procedure, or fault response."},
]
_CORPUS_PROVENANCE.update({
    "DIST-001": {"sources":["ISA_18_2","OSHA_PSM"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "DIST-002": {"sources":["OSHA_PSM"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "DIST-003": {"sources":["OSHA_PSM"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "DIST-004": {"sources":["OSHA_PSM","ISA_18_2"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "DIST-005": {"sources":["OSHA_PSM"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "DIST-006": {"sources":["OSHA_PSM"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "DIST-007": {"sources":["OSHA_PSM"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "DIST-008": {"sources":["OSHA_PSM","ISA_18_2"],"derivation":"constructed-illustrative","cleanliness":"C"},
    "DIST-009": {"sources":["OSHA_PSM","IEC_61882"],"derivation":"constructed-illustrative","cleanliness":"C"},
})
if os.environ.get("AXIOM_DISTRACTORS"):
    SOP_LIBRARY.extend(CS1_DISTRACTORS)   # re-run only; default OFF keeps the locked 51-chunk corpus

# Attach provenance to every corpus chunk IN PLACE. This adds an inert metadata key
# only; the RAG vectoriser indexes title+text(+keywords) ONLY (see RAGEngine.__init__),
# so provenance does NOT change retrieval, scores, or the cited-id set — the locked live
# study stays behaviour-identical. A chunk with no record gets an explicit MISSING marker
# so the enforcing test (tests/test_corpus_provenance_cs1.py) fails loudly rather than silently.
for _d in SOP_LIBRARY:
    _d["provenance"] = _CORPUS_PROVENANCE.get(
        _d["id"], {"sources": [], "derivation": "MISSING", "cleanliness": "MISSING"})

# Decision-critical knowledge sources: the operating-constraint SOPs that document the hard
# runaway limit (475 K) which EVERY coolant decision must respect. The certificate's evidence
# check requires the released proposal to GROUND in at least one of these whenever one was
# retrieved -- the CS1 analog of CS2's evidence coverage (e_min=1.0 over the decision-critical
# constraints). Mechanically derived from the corpus (drift fails the symmetry/coverage test).
EVIDENCE_E_MIN = 1.0
DECISION_CRITICAL_SOPS = [d["id"] for d in SOP_LIBRARY
                         if d.get("tier", "operating-constraint") == "operating-constraint"
                         and "475" in d["text"]]


# ── Lexical BM25 channel (symmetric with case2_flowsheet) ─────────────────────
# CS1 retrieval is a BM25 + TF-IDF (+ optional dense) hybrid fused by reciprocal
# rank fusion, identical in FORM to the column case study, so both case studies use
# the same cutting-edge hybrid retriever. Defined here so RAGEngine can build it.
import math as _math
from collections import Counter as _Counter

_RAG_STOP = None


def _rag_tokens(text):
    """Lowercase alphanumeric tokens, length >= 2, English stopwords removed (no
    stemming -- it hurts short domain terms like 'gas'/'tray'). Feeds the BM25 channel.
    Identical to case2_flowsheet._rag_tokens."""
    global _RAG_STOP
    if _RAG_STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _RAG_STOP = ENGLISH_STOP_WORDS
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(t) >= 2 and t not in _RAG_STOP]


# Conservative reactor-domain synonym map for QUERY expansion only (a recall booster;
# expanding the query, never the documents, keeps it low-risk). Mirrors CS2's design.
_RAG_SYNONYMS = {
    "cooling": ["coolant", "jacket"], "coolant": ["cooling"],
    "runaway": ["excursion", "thermal"], "temperature": ["thermal"],
    "feed": ["inlet"], "fault": ["failure", "anomaly"],
    "alarm": ["trip", "interlock"], "shutdown": ["trip", "sis"],
    "concentration": ["composition"], "heat": ["exotherm", "exothermic"],
    "loss": ["degradation", "reduction"], "setpoint": ["set-point"],
}


def _expand_query(query):
    """Tokenize + light domain synonym expansion (query side only)."""
    toks = _rag_tokens(query)
    extra = []
    for t in toks:
        extra.extend(_RAG_SYNONYMS.get(t, []))
    return toks + extra


class _BM25Lex:
    """Minimal Okapi BM25 over pre-tokenized documents (pure Python; no new deps).
    Identical formula to case2_flowsheet._BM25 (k1=1.5, b=0.75)."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b, self.docs = k1, b, docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = (sum(len(d) for d in docs_tokens) / self.N) if self.N else 0.0
        df, self.tf = {}, []
        for d in docs_tokens:
            counts = _Counter(d)
            self.tf.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: _math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
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
    HYBRID retriever, symmetric with the column case study (case2_flowsheet): a
    BM25 + TF-IDF (keyword-boosted) lexical hybrid, fused by reciprocal rank fusion
    (RRF, k=60), with an OPTIONAL real DENSE embedding channel (fastembed) fused into
    the same RRF -- a genuine sparse+dense hybrid. The dense channel is OFF by default
    and enabled with RAG_DENSE=1; see core/embeddings.py. `retrieve(method=...)` selects
    'hybrid' (default), 'bm25', or 'tfidf'. Each document carries source/version metadata
    for stale-knowledge tests and a tier ('operating-constraint' vs 'rationale') so the
    constraint and knowledge channels can be retrieved separately.
    """
    def __init__(self, docs, source="SOP_LIBRARY", version="1.0", use_dense=False):
        self.docs      = docs
        # Attach default provenance metadata (non-destructive) so retrieval
        # results can be checked for staleness.
        for d in self.docs:
            d.setdefault("source", source)
            d.setdefault("version", version)
            # Corpus tier (KB Phase B): "operating-constraint" (authoritative,
            # plant-specific — SOPs, design basis, limits) vs "rationale" (contextual
            # literature, NEVER a source of hard limits/feasibility). Legacy entries
            # default to operating-constraint; rationale refs carry an explicit tier.
            d.setdefault("tier", "operating-constraint")
        # Build combined text field (title + text + keywords if present)
        corpus = []
        for d in docs:
            kws = " ".join(d.get("keywords",[])) * 3   # boost keywords
            corpus.append(d["title"] + " " + d["text"] + " " + kws)
        # max_df=1.0: on a 25-document corpus a 0.9 cap can silently drop useful
        # terms that appear in most SOPs, so no document-frequency upper filter.
        self.vectorizer = TfidfVectorizer(stop_words="english",
                                          ngram_range=(1,2), max_df=1.0)
        self.matrix     = self.vectorizer.fit_transform(corpus)
        # BM25 lexical channel over the SAME keyword-boosted text, fused with the
        # TF-IDF cosine channel by RRF in retrieve() -- symmetric with case2_flowsheet,
        # so both case studies use the same BM25 + TF-IDF (+ dense) hybrid retriever.
        self._bm25 = _BM25Lex([_rag_tokens(t) for t in corpus])
        # Optional DENSE (embedding) channel for genuine sparse+dense hybrid retrieval.
        # OFF by default so the lexical study reproduces byte-for-byte; enable with
        # RAG_DENSE=1 (EMBED_BACKEND picks the model). Fails soft to lexical-only.
        self._dense = None
        self.dense_backend = "off (lexical-only)"
        if use_dense or os.environ.get("RAG_DENSE"):
            try:
                from case1_reactor.core.embeddings import DenseIndex
                di = DenseIndex([d["title"] + ". " + d["text"] for d in docs])
                self.dense_backend = di.backend_name
                if di.available:
                    self._dense = di
            except Exception as exc:
                diag_log.warning("dense RAG channel unavailable: %s", exc)
        # Optional cross-encoder reranker (RAG_RERANK=1): re-ranks the fused top-N more
        # precisely than the fused channels; fails soft to the fused order. Symmetric
        # with case2_flowsheet.
        self._reranker = None
        self.rerank_backend = "off"
        if os.environ.get("RAG_RERANK"):
            try:
                from case1_reactor.core.embeddings import CrossEncoderReranker
                rr = CrossEncoderReranker()
                self.rerank_backend = rr.backend_name
                if rr.available:
                    self._reranker = rr
            except Exception as exc:
                diag_log.warning("reranker unavailable: %s", exc)

    def retrieve(self, query, top_k=4, tiers=("operating-constraint",), method="hybrid"):
        # tiers restricts retrieval to the requested corpus tier(s). The DEFAULT is the
        # authoritative operating-constraint channel, so the rationale/literature tier is
        # never returned here and can never be treated as constraint-bearing (KB Phase B).
        # tiers=None retrieves across all tiers (e.g. an explicit rationale channel
        # retrieve(query, tiers=("rationale",))). Filtering the SAME argsort order keeps
        # the constraint channel's tie-breaking identical to the pre-enrichment behaviour.
        # method in {'hybrid','bm25','tfidf'} (symmetric with case2_flowsheet._rank):
        # 'hybrid' = RRF (k=60) of TF-IDF + BM25 [+ dense embeddings when RAG_DENSE=1].
        n_doc  = len(self.docs)
        tfidf  = cosine_similarity(self.vectorizer.transform([query]), self.matrix).flatten()
        if method == "tfidf":
            order, rank_score = np.argsort(tfidf)[::-1], tfidf
        else:
            bm = np.asarray(self._bm25.scores(_expand_query(query)), float)
            if method == "bm25":
                order, rank_score = np.argsort(bm)[::-1], bm
            else:
                # hybrid = TWO-LEVEL RRF: first fuse the two correlated lexical channels
                # (TF-IDF + BM25) into ONE lexical ranking, THEN fuse lexical + dense, so the
                # dense channel gets ~50% weight instead of being out-voted 2:1 by the two
                # redundant lexical rankings (the equal-weight 3-way fusion diluted dense's
                # strong top-rank, hurting MRR on the small reactor corpus). Channel weights
                # are configurable (RAG_W_LEX / RAG_W_DENSE) for weighted RRF. Symmetric with
                # case2_flowsheet._rank.
                def _rrf(orders, weights):
                    s = np.zeros(n_doc)
                    for o, w in zip(orders, weights):
                        rk = np.empty(n_doc, int); rk[o] = np.arange(n_doc)
                        s += w / (60 + rk + 1)
                    return s
                lex_rrf = _rrf([np.argsort(tfidf)[::-1], np.argsort(bm)[::-1]], [1.0, 1.0])
                if self._dense is not None:
                    dense_order = np.array(
                        [i for i, _s in self._dense.query(query, top_k=n_doc)], int)
                    w_lex   = float(os.environ.get("RAG_W_LEX", 1.0))
                    w_dense = float(os.environ.get("RAG_W_DENSE", 1.0))
                    rrf = _rrf([np.argsort(lex_rrf)[::-1], dense_order], [w_lex, w_dense])
                else:
                    rrf = lex_rrf
                order, rank_score = np.argsort(rrf)[::-1], rrf
        # tier filter, then OPTIONAL cross-encoder rerank of the fused top-N pool
        cand = [i for i in order
                if tiers is None
                or self.docs[i].get("tier", "operating-constraint") in tiers]
        if self._reranker is not None and cand:
            pool   = cand[:max(int(top_k), 20)]
            scores = self._reranker.rerank(query, [self.docs[i]["text"] for i in pool])
            if scores is not None:
                pool = [p for _, p in sorted(zip(scores, pool), key=lambda t: -t[0])]
                cand = pool + [i for i in cand if i not in pool]
        idx = cand[:top_k]
        return [{"id":self.docs[i]["id"], "title":self.docs[i]["title"],
                 "text":self.docs[i]["text"].strip(),
                 "tier":self.docs[i].get("tier", "operating-constraint"),
                 "source":self.docs[i].get("source"),
                 "version":self.docs[i].get("version"),
                 "score":round(float(rank_score[i]),4)} for i in idx]

    def full_matrix(self, queries):
        result = []
        for q in queries:
            qv = self.vectorizer.transform([q])
            sc = cosine_similarity(qv, self.matrix).flatten()
            result.append(sc)
        return np.array(result)


# ══════════════════════════════════════════════════════════════════════════════
# MCP-STYLE (IN-PROCESS) TOOL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class MCPToolRegistry(_CoreMCPToolRegistry):
    """Case study 1's tool registry = the shared core registry, keeping the
    original args-log truncation length (100 chars). Behaviour-identical to the
    previous inline class; see core/tools.py and RECONCILIATION.md."""
    def __init__(self):
        super().__init__(args_maxlen=100)


# (Removed the unused make_bo_ctrl / make_sis_ctrl / make_pid_ctrl factories:
#  the agent uses CounterfactualEngine.{grid_search_agent,sis_agent,pid_agent}
#  exclusively, so these never-called duplicates could only drift out of sync
#  with the executed comparators. See FIXES.md 2.2 / 4.4.)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS (10 scenarios)
# ══════════════════════════════════════════════════════════════════════════════

def build_scenarios(cstr):
    ss = cstr.steady_states()
    print("  [steady-states] found %d: %s" % (
        len(ss), ", ".join(f"T={s['T']:.1f}K({'stable' if s['stable'] else 'unstable'})"
                           for s in ss)))
    lss = cstr.lower_stable_ss()          # lower stable SS by stability, not index
    Ca_nom, T_nom = lss["Ca"], lss["T"]

    # Initial conditions represent an ABNORMAL EVENT ALREADY IN PROGRESS when the
    # agent engages. The runaway-capable model self-limits from nominal (~326 K), so
    # the hazardous scenarios start MID-FAULT in the alarm band (T~452 K) with the
    # in-situ reactant LARGELY CONSUMED (Ca~0.10) — realistic for a fault that has
    # been developing. Two honest physical facts shape the design:
    #  (1) Pure cooling loss at NOMINAL feed is REACTANT-LIMITED (genuine runaway
    #      needs reactant), so the cooling scenario carries elevated feed.
    #  (2) A high in-situ reactant charge (Ca>~0.3) at T>=450 ignites in a fast
    #      transient that NO cooling can stop (past the controllable point) — so the
    #      RECOVERABLE scenarios use low Ca, where aggressive cooling wins the
    #      kinetic race while under-cooling lets the feed re-drive runaway.
    # ICs were verified with DENSE-OUTPUT peaks (the default 2-point simulate samples
    # only endpoints and steps over the fast burn transient — see NOTES.md) so that,
    # in the closed-loop run, an insufficient action crosses T_run while the
    # aggressive action recovers; the safety filter and the gate's safety branch fire
    # during the run, not only in an isolated probe. S06 is deliberately PAST the
    # controllable point (even max cooling peaks > T_run) to exercise the fail-closed
    # safety branch.
    return [
        {"id":"S01","name":"Cooling Degradation (mid-fault)",
         "y0":[0.10, 452.0], "Ca0f":1.3, "UAf":0.50,
         "q_frac":1.0, "noise":0.3,
         "fault_onset":0, "fault_type":"cooling",
         "desc":"Cooling loss (UA 50%) at elevated feed load, event in progress: "
                "under-cooling (Tc>=345) crosses T_run (~477 K) while the verified "
                "controller's aggressive action holds it (closed-loop peak ~461 K, "
                "certificate safe). The safety screen rejects the under-cooling "
                "candidates and selects the safe action during the run."},

        {"id":"S02","name":"Feed Concentration Surge (mid-fault)",
         "y0":[0.10, 452.0], "Ca0f":2.0, "UAf":1.0,
         "q_frac":1.0, "noise":0.3,
         "fault_onset":0, "fault_type":"feed",
         "desc":"Ca0 surges to 2.0 mol/L (upstream mixing failure), event in "
                "progress: under-cooling runs away (~497 K) while the verified "
                "controller's aggressive cooling recovers (closed-loop peak "
                "~453 K, certificate safe)."},

        {"id":"S03","name":"Compound: Cooling Loss + Feed Surge (mid-fault)",
         "y0":[0.10, 452.0], "Ca0f":1.4, "UAf":0.55,
         "q_frac":1.0, "noise":0.3,
         "fault_onset":0, "fault_type":"compound",
         "desc":"Simultaneous partial cooling failure and feed surge in progress — "
                "the highest-risk single-failure combination; under-cooling crosses "
                "T_run (~482 K), the verified aggressive action recovers (closed-loop "
                "peak ~459 K, certificate safe)."},

        {"id":"S04","name":"Thermocouple Sensor Drift",
         "y0":[Ca_nom, T_nom+15], "Ca0f":1.0, "UAf":1.0,
         "q_frac":1.0, "noise":3.5,
         "fault_onset":0, "fault_type":"sensor",
         "desc":"High measurement noise (3.5 K std) — thermocouple drift; tests "
                "diagnosis robustness, not runaway."},

        {"id":"S05","name":"Incipient Cooling Degradation",
         "y0":[0.97, 335.0], "Ca0f":1.0, "UAf":0.82,
         "q_frac":1.0, "noise":0.8,
         "fault_onset":0, "fault_type":"cooling_incipient",
         "desc":"Mild cooling loss (UA 82%) — early/incipient degradation. The "
                "system is BISTABLE (low ~326 K or ignited >450 K); mild cooling "
                "stays on the LOW branch (settles ~329 K), so this is detected via "
                "the UA estimate, not a T alarm — early detection of a runaway "
                "precursor (relabeled from the prior oscillatory scenario; NOTES.md)."},

        {"id":"S06","name":"Near-Runaway Emergency",
         "y0":[0.30, 452.0], "Ca0f":1.4, "UAf":0.50,
         "q_frac":1.0, "noise":0.2,
         "fault_onset":0, "fault_type":"runaway",
         "desc":"Reactor PAST the controllable point (high in-situ reactant at the "
                "alarm band): even maximum cooling peaks > T_run (~485 K), so the "
                "certificate's safety branch fails and the system FAILS CLOSED rather "
                "than presenting an action it cannot vouch for. Exercises the "
                "fail-closed safety branch on an unrecoverable transient."},

        {"id":"S07","name":"Feed Flow Reduction",
         "y0":[Ca_nom, T_nom+12], "Ca0f":1.0, "UAf":1.0,
         "q_frac":0.58, "noise":0.3,
         "fault_onset":0, "fault_type":"flow",
         "desc":"Feed flow drops to 58% — pump degradation. q is UNOBSERVABLE "
                "(confounded with UA): the known-negative; the gate must flag/abstain."},

        {"id":"S08","name":"Cascading Multi-Fault (mid-fault)",
         "y0":[0.10, 452.0], "Ca0f":1.35, "UAf":0.50,
         "q_frac":0.75, "noise":1.5,
         "fault_onset":0, "fault_type":"cascade",
         "desc":"Cooling + feed + flow + sensor noise — cascading failure in progress."},

        {"id":"S09","name":"Time-Varying Fault: Delayed Cooling Loss",
         "y0":[Ca_nom, T_nom+8], "Ca0f":1.2, "UAf":0.95,
         "q_frac":1.0, "noise":0.4,
         "fault_onset":5, "fault_type":"time_varying",
         "desc":"Cooling degrades progressively from t=5 min at elevated feed load "
                "(developing onset; the time-varying injection is wired in P8)."},

        {"id":"S10","name":"Recovery Validation: Correct vs Incorrect Action",
         "y0":[0.10, 453.0], "Ca0f":1.3, "UAf":0.50,
         "q_frac":1.0, "noise":0.2,
         "fault_onset":0, "fault_type":"validation",
         "desc":"Near runaway: tests that the agent avoids the WRONG action (raising "
                "Tc, which runs away) and takes aggressive cooling."},
    ]


# ══════════════════════════════════════════════════════════════════════════════
# FAIL-CLOSED ADMISSIBILITY CERTIFICATE (Phase 1.3)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AdmissibilityCertificate:
    """
    Deterministic fail-closed gate over an AXIOM decision. The released decision —
    the verified safe action PLUS its SOP-grounded explanation — is presented as
    authoritative ONLY if every check holds; otherwise the system fails closed,
    commits the verified safe action, and flags the LLM narrative non-authoritative.
    Safety lives on the model-owned limit T_run and these checks, never on the LLM.

    P5: action_consistent (did the LLM's proposal match the executed action) is
    DROPPED — the LLM is a narrator, its proposal need not match — and REPLACED by
    explanation_consistent (does the released explanation truthfully state the
    verified action), which governs the narrative. robustness is added (P6 fills it).
    """
    converged: bool            # closed-loop integration produced finite states
    safe: bool                 # closed-loop peak under the executed action clears T_run by SAFE_MARGIN
    explanation_consistent: bool  # released explanation STATES the verified action (parsed == exec_tc within TC_TOL)
    evidence_cited: bool        # every SOP cited in the explanation was actually retrieved (no hallucinated/stale cite)
    margins: dict              # e.g. {"T_run_margin", "rec_T_run_margin"}
    robustness: bool = True     # action stays safe across the MC ensemble (P6 fills; stub pass-through until then)
    diagnosis_confident: bool = True  # P7: EnKF calibration >= threshold — the modelled
                                      # faults can explain the data (an unobservable fault
                                      # collapses calibration -> low-confidence diagnosis)
    recommendation_safe: bool = True  # the action STATED in the explanation, simulated under the EnKF
                                      # worst-case plant, clears T_run by SAFE_MARGIN
    recommended_tc: float = None      # the coolant setpoint PARSED from the explanation text
    executed_tc: float = None
    reason: str = ""


def passes(cert):
    """
    Admissible iff every genuine safety/governance check holds: convergence; the
    executed action is SAFE (closed-loop peak clears T_run by SAFE_MARGIN); it is
    ROBUST across the MC ensemble (P6); the released explanation truthfully states
    the verified action (EXPLANATION-CONSISTENT — governs the narrative); the cited
    evidence was actually RETRIEVED; the explanation's stated action is itself
    worst-case safe (recommendation_safe); and the DIAGNOSIS is confident (the EnKF
    can explain the data — an unobservable/misspecified fault collapses calibration
    and is NOT endorsed, P7). The diagnosis term governs the narrative, not the
    action: the safe action is still committed, but a low-confidence fault ID is not
    presented as authoritative.
    """
    return (cert.converged and cert.safe and cert.robustness and
            cert.diagnosis_confident and cert.explanation_consistent and
            cert.evidence_cited and cert.recommendation_safe)


def parse_recommended_tc(synthesis, tc_band=(240.0, 360.0)):
    """
    Extract the LLM-recommended coolant setpoint Tc (K) from the ACTION section,
    robustly to real-model prose.

    The setpoint is identified by an explicit COOLANT-SETPOINT anchor (the variable
    ``Tc``, ``coolant ...``, ``jacket ...``, or ``setpoint``) gated to a 3-digit Kelvin
    value. Narrative or HAZARD temperatures ("peak T may reach 480 K", "runaway above
    485 K") are NOT anchored to a coolant term and are ignored. (P12 hardening: the
    earlier ``T[c_]?\\w*`` pattern matched ANY word starting with T — "The 480 K ..." —
    and the bare-number fallback grabbed the first 3-digit Kelvin value, i.e. a hazard
    temperature; both fed a WRONG Tc to the certificate. Verified on real-prose-shaped
    input; the keyed smoke test validates it on actual model output.)

    FAILS CLOSED (returns None) when no coolant-setpoint phrase is found, or when a bare
    candidate is ambiguous or outside the physically plausible coolant band. A None
    leaves explanation_consistent / recommendation_safe UNCONFIRMABLE, so the certificate
    gate ABSTAINS (fails closed) rather than trusting a misread — extraction is never
    loosened to force a parse, and no safety check is relaxed to accommodate prose.
    An explicit anchored setpoint is returned even if out of band (it is the model's real
    recommendation; the safety simulation and consistency check then judge it — that is
    the honest recommendation_safe measurement). Returns float or None.
    """
    if not synthesis:
        return None
    m = re.search(r"ACTION:(.*?)(?:RISK:|$)", synthesis, re.S | re.I)
    section = m.group(1) if m else synthesis
    lo, hi = tc_band
    # Primary: an explicit coolant-setpoint anchor -> trust the model's stated value.
    anchor = r"(?:\bTc\b|\b(?:coolant|jacket)\b(?:\s+\w+){0,2}|\bset[\s-]?point\b)"
    m2 = re.search(anchor + r"\s*(?:to|=|:|at|of|near|around|->)?\s*"
                   r"([0-9]{3}(?:\.[0-9]+)?)\s*K\b", section, re.I)
    if m2:
        return float(m2.group(1))
    # Fallback: a bare 'NNN K' is trusted ONLY if exactly ONE distinct value lies in the
    # plausible coolant band (else ambiguous, or a hazard temperature -> fail closed).
    inband = sorted({float(x)
                     for x in re.findall(r"\b([0-9]{3}(?:\.[0-9]+)?)\s*K\b", section)
                     if lo <= float(x) <= hi})
    return inband[0] if len(inband) == 1 else None


# ══════════════════════════════════════════════════════════════════════════════
# MOCK LLM SYNTHESIS (offline / no-API path)
# ══════════════════════════════════════════════════════════════════════════════

def mock_synthesis(sop_ids, rec_tc=285.0):
    """
    Deterministic, well-formed stand-in for the Anthropic synthesis used when
    AXIOM_MOCK_LLM is set. Emits the four required sections (DIAGNOSIS,
    ATTRIBUTION, ACTION, RISK), cites the top retrieved SOP id, and states a
    fixed recommended coolant setpoint (rec_tc) so the offline path exercises
    the admissibility/consistency gate without spending API credits.
    """
    sid = sop_ids[0] if sop_ids else "SOP-002"
    # Ground the narrative in a decision-critical hard-limit SOP if one was retrieved (a proper
    # narrative cites the documented runaway limit it must respect) -> satisfies the gate's
    # decision-critical evidence-coverage check. If none was retrieved, the top SOP is cited.
    crit = next((s for s in sop_ids if s in set(DECISION_CRITICAL_SOPS)), None)
    cite = sid + (" and %s" % crit if crit and crit != sid else "")
    return (
        "DIAGNOSIS: MAP fault hypothesis is partial cooling loss (P=0.42). "
        "Degraded UA lowers jacket heat removal while Arrhenius-accelerated "
        "kinetics sustain heat generation, driving T up the upper steady-state "
        "branch.\n"
        "ATTRIBUTION: Sensitivity scores attribute the deviation primarily to "
        "UA (cooling); reduced UA shrinks the (UA/rho_Cp*V)*(T-Tc) removal term "
        "in the energy balance.\n"
        f"ACTION: (1) Set Tc to {rec_tc:.0f} K. (2) Reduce Ca0 feed by 25% to "
        f"0.75 mol/L. (3) Increase coolant flow by 30%. Per {cite}.\n"
        "RISK: Without action within 5 minutes, peak T may exceed 480 K with "
        "an estimated time-to-runaway under 3 minutes."
    )


# ══════════════════════════════════════════════════════════════════════════════
# OPTION-3 DECISION SPEC — the LLM's UPSTREAM structured proposal (planner role)
# ══════════════════════════════════════════════════════════════════════════════
#
# Under the re-architecture the LLM emits a schema-validated decision SPEC (not
# prose) BEFORE the action is computed. The spec can only NARROW/INFORM the
# authoritative search, never WIDEN the safety envelope:
#   hypothesis_weighting  {hyp_key: w>=0}  -> seeds the Bayesian PRIOR; the tracker
#                         still updates it with data, so a bad prior is corrected
#                         (at worst it costs diagnostic efficiency, never safety).
#   candidate_actions     [Tc_K, ...]      -> candidate coolant setpoints the IT
#                         selector evaluates; EACH is still safety-screened on the
#                         estimated worst-case plant against the AUTHORITATIVE,
#                         fixed cstr.T_run, and the selector keeps an authoritative
#                         safe fallback, so a proposed unsafe candidate is rejected.
#   contextual_constraints {operating_envelope, procedure, citations} -> gate
#                         EVIDENCE + SOFT/advisory constraints ONLY. The HARD
#                         runaway limit and trips live in CSTRModel (T_run/T_alarm),
#                         never in the spec — a stale/loosened proposed limit cannot
#                         change them. The safety guarantee never depends on the LLM.

def validate_decision_spec(spec, valid_hyps, valid_sops, tc_bounds=(240.0, 360.0),
                           require_citations=True):
    """Schema-validate an LLM decision spec. Returns (ok, errors). A malformed spec
    is REJECTED (never coerced); the caller fails closed to a fixed default spec.

    require_citations=False is for the +MCP-only ablation arm, whose prompt MANDATES an
    empty citation list (it has no retrieval tool) -- requiring citations there would
    auto-reject every spec and silently turn the arm into no-spec. Any citations that ARE
    present are still validated against the retrieved set (default behavior unchanged)."""
    errs = []
    if not isinstance(spec, dict):
        return False, ["spec is not a dict"]

    hw = spec.get("hypothesis_weighting")
    if not isinstance(hw, dict) or not hw:
        errs.append("hypothesis_weighting missing or empty")
    else:
        for k, v in hw.items():
            if k not in valid_hyps:
                errs.append(f"unknown hypothesis '{k}'")
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                errs.append(f"weight for '{k}' must be a non-negative number")
        nums = [float(v) for v in hw.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if sum(nums) <= 0:
            errs.append("hypothesis_weighting sums to <= 0")

    ca = spec.get("candidate_actions")
    lo, hi = tc_bounds
    if not isinstance(ca, (list, tuple)) or not ca:
        errs.append("candidate_actions missing or empty")
    else:
        for a in ca:
            if not isinstance(a, (int, float)) or isinstance(a, bool):
                errs.append(f"candidate action '{a}' is not numeric")
            elif not (lo <= float(a) <= hi):
                errs.append(f"candidate action {a} K outside plausible bounds {tc_bounds}")

    cc = spec.get("contextual_constraints")
    if not isinstance(cc, dict):
        errs.append("contextual_constraints missing")
    else:
        cites = cc.get("citations")
        if not isinstance(cites, (list, tuple)) or not cites:
            if require_citations:
                errs.append("contextual_constraints.citations missing or empty")
        else:
            for c in cites:
                if c not in valid_sops:
                    errs.append(f"citation '{c}' is not a known/retrieved SOP id")
    return (len(errs) == 0), errs


def mock_decision_spec(top_hyps, sop_ids, safety_critical=False):
    """Realistic, schema-VALID mock spec (AXIOM_MOCK_LLM) so mock runs are
    meaningful (Option 3 makes the live run load-bearing; the mock must still
    emit a sensible prior, plausible candidate setpoints, and real SOP citations).
    NOTE: the advisory operating_envelope here is SOFT; the hard runaway limit is
    NOT carried in the spec — it stays authoritative in CSTRModel."""
    hw = {}
    for i, item in enumerate(top_hyps[:3]):
        hkey, p = item[0], item[1]
        hw[hkey] = round(max(float(p), 0.05) * (1.0 - 0.2 * i), 4)
    if not hw:
        hw = {"H1_normal": 1.0}
    cands = ([255.0, 270.0, 285.0, 300.0] if safety_critical
             else [275.0, 290.0, 300.0, 315.0, 330.0])
    cites = list(sop_ids[:2]) or ["SOP-002"]
    return {
        "hypothesis_weighting": hw,
        "candidate_actions": cands,
        "contextual_constraints": {
            "operating_envelope": {"T_alarm_K": 450.0, "T_max_advisory_K": 460.0},
            "procedure": "stabilize temperature and restore cooling margin",
            "citations": cites,
        },
    }


# ── LIVE proposer (P1): a strict drop-in for mock_decision_spec ──────────────────────────
# The LIVE proposer ONLY replaces the function that PRODUCES the spec; the downstream
# consumption (validate -> prior seeding -> candidate augmentation, axiom.py ~2416-2437) is
# untouched. The proposal can only NARROW (candidates clipped+union-ed to [250,345]); it cannot
# alter T_run or the certificate (P0). On any parse/validation failure the parser returns {},
# which validate_decision_spec REJECTS, triggering the existing conservative fallback (None,None
# -> static prior + grid candidates). Prompt frozen in LIVE_NULL_SPEC.md (P3).
LIVE_PROPOSER_PROMPT = """You are the planning layer of a safety-critical CSTR decision-support \
system. From the evidence below, return ONLY a JSON object (no prose, no markdown fence) with \
exactly these keys:
  "hypothesis_weighting": an object mapping fault-hypothesis keys (from the provided list ONLY) \
to non-negative weights expressing your prior belief;
  "candidate_actions": a list of advisory coolant setpoints in Kelvin (numbers) to consider;
  "contextual_constraints": an object with "operating_envelope" (advisory: the key numeric \
temperature limits in Kelvin you are relying on -- e.g. the alarm/advisory thresholds -- taken ONLY \
from the retrieved SOPs; do NOT invent limit values), a short "procedure" string, and "citations" \
(a list of SOP ids drawn ONLY from the retrieved set below).
The HARD runaway limit and all safety screening are enforced DOWNSTREAM by the process model and \
are NOT yours to set; your proposal can only inform, never override, the verified controller.

QUERY: {query}
ENKF STATE: {enkf}
FAULT HYPOTHESES (use these keys only): {hyps}
RETRIEVED SOPs (cite ids only from here):
{sops}
Return the JSON object now."""


def parse_decision_spec_json(raw):
    """Extract a decision-spec JSON object from raw LLM text (strip prose/markdown), and
    normalize hypothesis_weighting to the simplex. Returns {} on ANY failure so that
    validate_decision_spec rejects it and the existing conservative fallback fires. Never
    coerces a malformed spec into a 'valid' one."""
    if not raw:
        return {}
    m = re.search(r"\{.*\}", raw, re.S)            # first balanced-ish {...} block
    if not m:
        return {}
    try:
        spec = json.loads(m.group(0))
    except Exception:
        return {}
    if not isinstance(spec, dict):
        return {}
    hw = spec.get("hypothesis_weighting")
    if isinstance(hw, dict) and hw:                # normalize w to the simplex (>=0, sum 1)
        nums = {k: float(v) for k, v in hw.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0}
        s = sum(nums.values())
        if s > 0:
            spec["hypothesis_weighting"] = {k: round(v / s, 6) for k, v in nums.items()}
    return spec


def live_decision_spec(client, model, query, retrieved, top_hyps, enkf_summary, safety_critical):
    """Call the pinned live model for a RAG-grounded decision spec. Returns (raw, parsed, model_id).
    Same schema as mock_decision_spec; parsing/validation handled by parse_decision_spec_json +
    the existing validate_decision_spec downstream."""
    hyp_keys = [h[0] for h in top_hyps]
    sop_block = "\n".join("%s — %s" % (r["id"], (r.get("text", "") or "").strip()[:240])
                          for r in retrieved) or "(none)"
    prompt = LIVE_PROPOSER_PROMPT.format(
        query=query, enkf=enkf_summary, hyps=", ".join(hyp_keys), sops=sop_block)
    # AXIOM_ONESHOT_MAX_TOKENS: per-run override (default 600 = the locked-study cap). The re-run's
    # one-shot arms (A0 bare / A1r +RAG) request the operating_envelope limits too, so they get
    # headroom (1500) -- a truncated spec would count as contamination, not as a model failure.
    resp = client.messages.create(model=model,
                                  max_tokens=int(os.environ.get("AXIOM_ONESHOT_MAX_TOKENS", "600")),
                                  temperature=0.0,
                                  messages=[{"role": "user", "content": prompt}])
    raw = resp.content[0].text
    return raw, parse_decision_spec_json(raw), getattr(resp, "model", model)


def _log_live(kind, payload):
    """Append a verbatim live-LLM record (model id + ISO-UTC date) to results/live/live_log.jsonl."""
    import datetime
    d = os.path.join(RESULT_DIR, "live")
    os.makedirs(d, exist_ok=True)
    rec = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(), "kind": kind, **payload}
    with open(os.path.join(d, "live_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# Optional scenario/rep context merged into the live "proposer" log entry by run_live's null arm, so
# every raw proposer response is self-identifying (per-scenario checks need no order/query inference).
# Empty by default -> a proposer call outside the live eval logs exactly as before (backward compatible).
LIVE_PROPOSER_TAG = {}


def _assert_response_contract(resp):
    """Assert the live SDK response exposes EXACTLY what the harness consumes, so a contract drift
    between the real SDK and the (fake-verified) harness fails the PRE-FLIGHT cleanly instead of
    crashing on the first scenario. The fake in the dry run was built to match these assumptions;
    this is the only check that exercises the REAL object. Checks resp.content[0].text (str),
    resp.model (str), resp.usage.input_tokens / output_tokens (int). Raises ValueError on mismatch."""
    try:
        text = resp.content[0].text
    except Exception as e:
        raise ValueError("response.content[0].text missing/invalid (%s)" % e)
    if not isinstance(text, str):
        raise ValueError("response.content[0].text is %s, expected str" % type(text).__name__)
    if not isinstance(getattr(resp, "model", None), str):
        raise ValueError("response.model missing or not a str")
    u = getattr(resp, "usage", None)
    if u is None:
        raise ValueError("response.usage missing")
    for attr in ("input_tokens", "output_tokens"):
        v = getattr(u, attr, None)
        if not isinstance(v, int):
            raise ValueError("response.usage.%s missing or not an int (got %r)" % (attr, v))
    return True


def preflight_model_check(model=LLM_MODEL, api_key=None, client=None, log=True):
    """P4 STEP 0 — model-resolution pre-flight (gate before any run budget is spent).

    Confirm the pinned model string (LLM_MODEL, axiom.py:72) actually RESOLVES for THIS account,
    so a bad/inaccessible model fails fast rather than mid-run. Returns a status dict; it never
    fabricates a resolution.

    - **No key** (and no injected client): degrades cleanly — reports the key is absent and returns
      `status="pending"` with `resolved=None`. It does NOT call the API and does NOT invent a
      resolved id. P4 is gated on the key elsewhere; this just makes the check safe to call offline.
    - **Key present:** (a) best-effort enumerates the account's available model ids via
      `client.models.list()` (logged if the endpoint is reachable); (b) makes ONE minimal call
      (`max_tokens=1`) with the pinned string — the authoritative check that the account can invoke
      it — and reads back the RESOLVED id (`resp.model`), the ISO-UTC date, and `resp.usage` (the
      real token counts, used to reconfirm the a-priori cost estimate). On success returns
      `status="ok"` with the resolved id. On any failure (model not available to the account, auth
      error) returns `status="error"` with the reason; the caller (run_live) ABORTS on non-ok.

    `client` may be injected for testing (a stub with `.messages.create`/`.models.list`)."""
    import datetime
    api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # No key and no injected client -> pending; never fabricate a resolution.
    if not api_key and client is None:
        result = {"status": "pending", "model": model, "resolved": None,
                  "available": None, "date": now_iso,
                  "reason": "ANTHROPIC_API_KEY not set"}
        print("  [PREFLIGHT] ANTHROPIC_API_KEY not set -> model resolution PENDING "
              "(not fabricated). Pinned model stays '%s'." % model)
        if log:
            _log_live("preflight", result)
        return result

    try:
        client = client if client is not None else anthropic.Anthropic(api_key=api_key)
        # (a) best-effort: list available models (no generation tokens consumed).
        available = None
        try:
            available = [m.id for m in client.models.list(limit=1000).data]
        except Exception as err:
            diag_log.warning("preflight: models.list unavailable (%s); relying on the "
                             "minimal call to resolve the model", err)
        # (b) authoritative: one MINIMAL call with the pinned string; read back the resolved
        #     id (resp.model) + usage (real token counts -> reconfirms LIVE_RUN_COST_ESTIMATE).
        #     max_tokens is 16 (not 1): a 1-token cap stops at stop_reason=max_tokens BEFORE any
        #     content block is emitted (resp.content == []), which fails _assert_response_contract
        #     spuriously; 16 reliably yields one TextBlock so the contract is exercised on real
        #     output (verified empirically: max_tokens=1 -> 0 blocks; max_tokens=16 -> 1 TextBlock).
        resp = client.messages.create(
            model=model, max_tokens=16, temperature=0.0,
            messages=[{"role": "user", "content": "ping"}])
        # (A) REAL SDK contract gate: the fake cannot prove this -- assert the live response shape
        # the harness consumes, so a drift aborts the pre-flight rather than crashing mid-run.
        _assert_response_contract(resp)
        resolved = getattr(resp, "model", None) or model
        usage = getattr(resp, "usage", None)
        result = {"status": "ok", "model": model, "resolved": resolved,
                  "available": available, "date": now_iso,
                  "contract_verified_on_first_call": True,
                  "usage": ({"input_tokens": getattr(usage, "input_tokens", None),
                             "output_tokens": getattr(usage, "output_tokens", None)}
                            if usage is not None else None)}
        print("  [PREFLIGHT] OK: pinned '%s' resolved to '%s' (%s)." % (model, resolved, now_iso))
        print("  [PREFLIGHT] (1C) SDK response contract (content[0].text, model, "
              "usage.input_tokens/output_tokens) verified EMPIRICALLY on THIS first real call -- "
              "before now it was only asserted against a fake of the same shape. Confirm the first "
              "few logged calls show real integer token usage before trusting the full set.")
        if log:
            _log_live("preflight", result)
        return result
    except Exception as err:
        result = {"status": "error", "model": model, "resolved": None,
                  "available": None, "date": now_iso,
                  "reason": "%s: %s" % (type(err).__name__, err)}
        print("  [PREFLIGHT] FAILED for pinned model '%s' (%s) -> ABORT, do NOT run."
              % (model, err))
        if log:
            _log_live("preflight", result)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# AXIOM AGENT — FULL AGENTIC REASONING LOOP
# ══════════════════════════════════════════════════════════════════════════════

class AXIOMAgentV2:
    """
    AXIOM v2 full agentic loop integrating all five pillars:
    P1: CSTR model + anomaly detection
    P2: EnKF state estimation
    P3: Bayesian hypothesis belief tracking
    P4: Information-theoretic action selection
    P5: Causal attribution scoring
    + LLM synthesis via Anthropic API
    + MCP-style (in-process) tool registry
    """

    def __init__(self, api_key, cstr, rag, registry, mock=False, model=LLM_MODEL):
        # Client is created lazily so that (a) importing/constructing never
        # requires a key, and (b) a missing key raises a clear, actionable
        # error only when a live LLM call is actually attempted.
        self.api_key  = api_key
        self.mock     = mock
        self.model    = model
        self._client  = None
        self.cstr     = cstr
        self.rag      = rag
        self.registry = registry
        self.results  = []
        # LLM resilience (Phase 4.2): bounded retries with exponential backoff
        # and a per-call timeout. retry_backoff is the base; tests set it to 0.
        self.max_retries   = 3
        self.retry_backoff = 2.0
        self.llm_timeout   = 60.0

    def _synthesize(self, user_msg, sops, verified_tc=None):
        """Return (synthesis_text, model_id). Mock path is deterministic; live
        path retries transient API errors with exponential backoff and a
        timeout, raising only after max_retries failures (Phase 4.2).

        verified_tc (P5): the verified committed action. The mock explanation STATES
        it (so a truthful explanation passes explanation-consistency); the live LLM
        is already given it in the prompt and is expected to state it. The
        explanation-consistency check (certificate) re-parses the text and compares,
        so a text that MISSTATES the action still fails closed — the check is a
        genuine parse-and-compare, not tautological."""
        if self.mock:
            print("  [LLM] MOCK mode — deterministic synthesis (no API call).")
            rec = verified_tc if verified_tc is not None else 285.0
            return mock_synthesis([r["id"] for r in sops], rec_tc=rec), "mock"
        client   = self._client_or_raise()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(
                    model=self.model, max_tokens=3072, timeout=self.llm_timeout,
                    temperature=0.0,   # greedy decoding for reproducible live results
                    system=self.system_prompt(),                # (P12 pre-flight; the
                    messages=[{"role": "user", "content": user_msg}])  # API has no seed)
                return resp.content[0].text, getattr(resp, "model", self.model)
            except Exception as err:
                last_err = err
                wait = self.retry_backoff * (2 ** attempt)
                diag_log.warning("LLM attempt %d/%d failed: %s; retry in %.1fs",
                                 attempt + 1, self.max_retries, err, wait)
                print(f"  [LLM] attempt {attempt+1}/{self.max_retries} "
                      f"failed: {err}")
                if attempt < self.max_retries - 1 and wait > 0:
                    time.sleep(wait)
        raise RuntimeError(
            f"LLM synthesis failed after {self.max_retries} attempts: {last_err}")

    def _client_or_raise(self):
        """Lazily construct the Anthropic client; raise an actionable error
        if no key is available (only reached on the live path)."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Export it to use the live "
                    "LLM path, or set AXIOM_MOCK_LLM=1 to run the full pipeline "
                    "offline with a deterministic mock synthesis.")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _propose_spec(self, query, retrieved, top_hyps, safety_critical, enkf_summary="",
                      sim_state=None):
        """Produce the structured decision spec (P1, Option 3). MOCK -> mock_decision_spec
        (unchanged drop-in); LIVE -> live_decision_spec via the pinned model, logged verbatim.
        A live parse/validation failure returns {} so the existing conservative fallback
        (None,None -> static prior + grid candidates) fires. The downstream consumption is
        IDENTICAL for both sources -- proven byte-equal in P2a; only the spec content changes,
        never the plumbing.

        When AXIOM_TOOL_USE is set (live only), the proposer GROUNDS itself via the real MCP
        tools (retrieve_sops + simulate_setpoint) before returning the spec; sim_state =
        (Ca, T, UAf_screen, Ca0f_screen) supplies simulate_setpoint's worst-case context under
        the ESTIMATED state (the gate still re-verifies independently under the true state).
        Default OFF -> the one-shot path is untouched as the comparison arm."""
        sop_ids = [r["id"] for r in retrieved]
        if self.mock:
            return mock_decision_spec(top_hyps, sop_ids, safety_critical=safety_critical)
        client = self._client_or_raise()
        if os.environ.get("AXIOM_MCP_ONLY") and sim_state is not None:
            # +MCP-only ablation cell: physics tool (simulate_setpoint) ONLY, NO retrieval
            # tool -- isolates MCP's standalone contribution (the action hallucination family).
            from case1_reactor import mcp_server
            Ca, T, uaf_screen, ca0f_screen = sim_state
            spec, tool_log, raw_text = mcp_server.tooluse_decision_spec(
                client, self.model, query, top_hyps, enkf_summary,
                self.cstr, self.rag, Ca, T, uaf_screen, ca0f_screen, mcp_only=True)
            _log_live("proposer", {"model": self.model, "query": query,
                                   "raw": raw_text, "parsed_spec": spec,   # raw = VERBATIM model text
                                   "tool_use": True, "mcp_only": True, "tool_log": tool_log,
                                   "retrieved_ids": sop_ids,   # upstream retrieval (NOT shown to the model here)
                                   "enkf_summary": enkf_summary,   # completes prompt reconstruction
                                   "hyp_keys": [h[0] for h in top_hyps],
                                   **LIVE_PROPOSER_TAG})
            return spec
        if (os.environ.get("AXIOM_TOOL_USE") and sim_state is not None
                and not os.environ.get("AXIOM_NO_RAG")):   # bare arm (NO_RAG) stays one-shot, no tools
            from case1_reactor import mcp_server
            Ca, T, uaf_screen, ca0f_screen = sim_state
            spec, tool_log, raw_text = mcp_server.tooluse_decision_spec(
                client, self.model, query, top_hyps, enkf_summary,
                self.cstr, self.rag, Ca, T, uaf_screen, ca0f_screen)
            _log_live("proposer", {"model": self.model, "query": query,
                                   "raw": raw_text, "parsed_spec": spec,   # raw = VERBATIM model text
                                   "tool_use": True, "tool_log": tool_log,
                                   "retrieved_ids": sop_ids,
                                   "enkf_summary": enkf_summary,
                                   "hyp_keys": [h[0] for h in top_hyps],
                                   **LIVE_PROPOSER_TAG})
            return spec
        raw, spec, model_id = live_decision_spec(client, self.model, query, retrieved,
                                                 top_hyps, enkf_summary, safety_critical)
        # retrieved_ids makes the proposer record self-contained for per-arm grounding scoring
        # (cited subset-of retrieved) -- [] for the bare arm (retrieval stripped upstream).
        # enkf_summary + hyp_keys complete the verbatim-prompt reconstruction (the frozen
        # LIVE_PROPOSER_PROMPT template + these fields + the retrieved texts = the exact prompt).
        _log_live("proposer", {"model": model_id, "query": query, "raw": raw, "parsed_spec": spec,
                               "retrieved_ids": sop_ids, "enkf_summary": enkf_summary,
                               "hyp_keys": [h[0] for h in top_hyps], **LIVE_PROPOSER_TAG})
        return spec

    def system_prompt(self):
        return (
            "You are AXIOM v2, a PhD-level process safety agentic AI for an "
            "exothermic CSTR. You receive multi-source evidence: (1) EnKF-estimated "
            "process state with uncertainty bounds, (2) Bayesian posterior over 10 "
            "competing fault hypotheses with KL-divergence revision signal, "
            "(3) causal attribution scores quantifying root cause contributions, "
            "(4) the verified protective action computed by a conservative "
            "safety-screened controller (you do NOT choose the action; it is the "
            "most-conservative action the safety screen verifies), and "
            "(5) RAG-retrieved SOP context. "
            "Your response must have exactly four labeled sections:\n"
            "DIAGNOSIS: State the MAP fault hypothesis and its posterior probability. "
            "Give a 2-sentence mechanistic causal chain using process engineering terms "
            "(Arrhenius, heat balance, reaction kinetics, speciation, etc.).\n"
            "ATTRIBUTION: Interpret the causal attribution scores — which process "
            "variable is the primary root cause and by what mechanism?\n"
            "ACTION: Give exactly 3 quantitative corrective actions with specific "
            "numerical setpoints (K, mol/L, % flow). Reference the SOP ID.\n"
            "RISK: One sentence — consequence if no action in 5 minutes, with "
            "estimated peak T and time-to-runaway if calculable.\n"
            "Be precise, quantitative, and mechanistically rigorous."
        )

    def axiom_ctrl_fn(self, Ca0f, UAf, q_frac, belief, it_selector, enkf,
                      ts_log=None, rng=None, extra_candidates=None):
        """
        AXIOM closed-loop control function using the conservative safety-screened
        action selection. ``extra_candidates`` are the LLM decision-spec's proposed
        coolant setpoints (P3); they are merged into the screened candidate set but
        cannot change the committed action (the screen + conservative rule decide).

        If ``ts_log`` is provided, one record is appended at every control step
        capturing the full posterior over the 10 hypotheses, the EnKF ensemble
        mean/std for [Ca, T, UAf, Ca0f], the selected coolant setpoint, and the
        EIG of the selected action. This is the per-scenario closed-loop time
        series consumed by figures F3/F4 and persisted by export_results — the
        data the original code computed and then discarded.
        """
        gen = rng if rng is not None else np.random
        last_tc = [self.cstr.Tc_nom]   # previously EXECUTED coolant setpoint
        def fn(t, T, Ca):
            # EnKF predict under the ACTUAL last-executed action, not nominal Tc
            # (round 2, item 5): predicting under nominal cooling while the plant
            # is aggressively cooled biased the estimate by ~9 K and destroyed
            # calibration (mean coverage ~0.11); predicting under the executed
            # action restores coverage to ~0.95.
            enkf.predict(Tc=last_tc[0], dt=0.25)
            obs  = np.array([T + gen.normal(0, 1.5),
                             Ca + gen.normal(0, 0.01)])
            enkf.update(obs, t=t)
            es   = enkf.get_state()
            # Belief update
            belief.update(T, Ca, es, t=t)
            # conservative safety-screened action (screen on the estimated degraded plant)
            safety_crit   = T >= self.cstr.T_alarm
            uaf_s, ca0f_s = enkf.screen_params()
            it_act = it_selector.select(Ca, T, belief.posterior,
                                        belief, safety_crit,
                                        UAf_screen=uaf_s, Ca0f_screen=ca0f_s,
                                        extra_candidates=extra_candidates)
            if ts_log is not None:
                ts_log.append({
                    "t":         float(t),
                    "posterior": belief.posterior.copy(),
                    "enkf_mean": enkf.ensemble.mean(axis=0).copy(),
                    "enkf_std":  enkf.ensemble.std(axis=0).copy(),
                    "Tc":        float(it_act["Tc_selected"]),
                    "eig":       float(it_act["eig"]),
                })
            last_tc[0] = float(it_act["Tc_selected"])   # for the next predict
            return {"Tc": it_act["Tc_selected"], "q_frac": q_frac}
        return fn

    def run_scenario(self, scenario, t_sim=25.0, dt=0.25, n_mc=30, seed=None):
        sid   = scenario["id"]
        sname = scenario["name"]
        y0    = scenario["y0"]
        Ca0f  = scenario["Ca0f"]
        UAf   = scenario["UAf"]
        q_frac= scenario["q_frac"]
        noise = scenario["noise"]

        # P8: time-varying cooling fault. For a "time_varying" scenario the plant's
        # UAf RAMPS (make_uaf_ramp) from its pre-onset value to UAF_RAMP_END starting
        # at fault_onset. uaf_fn is passed to EVERY closed-loop comparator below
        # (AXIOM, the MC bands, and all baselines) so they run on the SAME developing
        # plant; uaf_fn=None means the constant-UAf plant (all other scenarios).
        uaf_fn = (make_uaf_ramp(UAf, UAF_RAMP_END, scenario["fault_onset"],
                                UAF_RAMP_DURATION)
                  if (scenario.get("fault_type") == "time_varying"
                      and not os.environ.get("AXIOM_NO_RAMP"))  # ablation (P8 check / P11)
                  else None)

        # Explicit per-scenario Generator (Phase 4.1): all randomness in this
        # scenario draws from rng, seeded deterministically from the scenario id,
        # so a scenario reproduces identically whether run alone or in the sweep.
        rng = np.random.default_rng(seed if seed is not None
                                    else scenario_seed(sid))

        print(f"\n{'═'*68}")
        print(f"  AXIOM v2 | {sid}: {sname}")
        print(f"  T0={y0[1]:.1f}K  Ca0={y0[0]:.3f}  Ca0f={Ca0f}  "
              f"UAf={UAf}  q={q_frac}  noise={noise}")
        print(f"{'═'*68}")

        t_total = time.time()

        # ── P2: Initialize EnKF
        enkf  = EnsembleKalmanFilter(self.cstr, N=80, rng=rng)
        enkf.ensemble = enkf._init_ensemble([y0[0], y0[1], UAf, Ca0f])

        # ── P3: Initialize Bayesian belief
        belief = HypothesisBelief(self.cstr, kl_threshold=0.18)

        # ── P4: Information-theoretic action selector
        it_sel = ConservativeSafetyScreenedController(self.cstr, n_candidates=25)

        # ── Initial anomaly detection
        dT_dt_est = None
        anomaly = self.registry.call("detect",
                                     T=y0[1], Ca=y0[0], UAf=UAf)

        # ── EnKF initial update
        obs0 = np.array([y0[1] + rng.normal(0,1.5),
                          y0[0] + rng.normal(0,0.01)])
        enkf.update(obs0, t=0.0)
        es0 = enkf.get_state()
        print(f"  EnKF init: T={es0['T']}±{es0['std_T']}K  "
              f"UAf={es0['UAf']:.3f}±{es0['std_UA']:.3f}")

        # ── P3: First belief update
        b0 = self.registry.call("belief_update",
                                 obs_T=y0[1], obs_Ca=y0[0],
                                 enkf_state=es0,
                                 belief=belief, t=0.0)
        top3 = belief.get_top_hypotheses(3)
        print(f"  MAP hypothesis: {top3[0][0]} "
              f"(P={top3[0][1]:.3f}) — {top3[0][2][:45]}")

        # ── P4: Information-theoretic action (safety screen on estimated plant)
        safety_crit     = y0[1] >= self.cstr.T_alarm
        uaf_s0, ca0f_s0 = enkf.screen_params()
        it_act = self.registry.call("it_action",
                                    Ca=y0[0], T=y0[1],
                                    posterior=belief.posterior,
                                    belief=belief,
                                    safety_crit=safety_crit,
                                    it_sel=it_sel,
                                    UAf_screen=uaf_s0, Ca0f_screen=ca0f_s0)
        print(f"  Conservative Tc: {it_act['Tc_selected']}K "
              f"(EIG={it_act['eig']:.5f})")

        # ── P5: Causal attribution — EXPLANATION-ONLY (finding 2). Scored AFTER the
        # action (it takes it_act's Tc as an input), `fp` flows only to the LLM
        # explanation prompt below and to the logged trace / F5 figure. It does NOT
        # feed the action, the certificate/gate, or the diagnosis; perturbing it
        # changes none of those (test_attribution_is_explanation_only_not_consumed).
        attr_scorer = CausalAttributionScorer(self.cstr)
        fp = self.registry.call("causal_attr",
                                Ca=y0[0], T=y0[1],
                                enkf_state=es0, Tc=it_act["Tc_selected"],
                                scorer=attr_scorer)
        print(f"  Causal fingerprint: {fp['top3'][:2]}")

        # ── RAG retrieval
        rag_q = (f"T={y0[1]}K Ca={y0[0]:.3f} Ca0f={Ca0f} UAf={UAf} "
                 f"events {' '.join(anomaly['events'])} "
                 f"hypothesis {top3[0][0]} {top3[0][2][:30]}")
        sops  = self.registry.call("retrieve_sop", query=rag_q, top_k=4)
        # Knowledge (rationale) channel, symmetric with the column's retrieve_knowledge:
        # the design-rationale (REF) tier is retrieved separately and UNIONED into the
        # retrieved evidence, so a rationale citation is grounded too (the gate's evidence
        # check accepts any retrieved id, exactly as CS2 accepts rationale + spec sources).
        refs  = self.rag.retrieve(rag_q, top_k=4, tiers=("rationale",))   # symmetric with CS2 retrieve_knowledge top_k
        if os.environ.get("AXIOM_NO_RAG"):
            # No-RAG ablation = the 'bare' three-stage measurement point. Strip retrieval
            # so the proposer and explainer must answer ungrounded; the evidence/citation
            # gate then measures the hallucination RAG prevents (symmetric to CS2's
            # plan(rag_on=False)). Defaults OFF, so the grounded study is unchanged.
            sops, refs = [], []
        seen_ids = {r["id"] for r in sops}
        sops = sops + [r for r in refs if r["id"] not in seen_ids]
        sop_ctx = "\n\n".join(
            f"[{r['id']}] {r['title']}:\n{r['text']}" for r in sops)

        # ── P3: LLM decision-spec — a GROUNDED PROPOSAL with NO authority. The LLM
        # proposes a hypothesis weighting (seeds the Bayesian prior) and candidate
        # coolant setpoints (merged into the screened candidate set); the data-driven
        # tracker and the worst-case safety screen remain authoritative. A malformed
        # or absent spec falls back to the fixed default (static prior + grid
        # candidates); validate_decision_spec REJECTS, never coerces. The proposal is
        # recorded but demonstrably changes neither the committed action (the screen +
        # conservative rule decide — finding 3) nor the final diagnosis (the prior
        # washes out under data — finding 4); see NOTES "P3". (Mock build: the spec is
        # the deterministic mock_decision_spec; parsing a live LLM spec is the
        # separate live step, like _synthesize.)
        sop_ids_ret   = [r["id"] for r in sops]
        decision_spec = self._propose_spec(rag_q, sops, top3, safety_crit,
                                            enkf_summary="T=%sK UAf=%s" % (es0["T"], es0["UAf"]),
                                            sim_state=(es0["Ca"], es0["T"], uaf_s0, ca0f_s0))
        spec_ok, spec_errs = validate_decision_spec(
            decision_spec, list(FAULT_HYPOTHESES.keys()),
            [s["id"] for s in SOP_LIBRARY],
            require_citations=not os.environ.get("AXIOM_MCP_ONLY"))
        if spec_ok:
            spec_prior, spec_cands = (decision_spec["hypothesis_weighting"],
                                      decision_spec["candidate_actions"])
        else:
            diag_log.warning("decision-spec REJECTED (%s); fixed default "
                             "(static prior + grid candidates)", spec_errs)
            spec_prior, spec_cands = None, None
        # Ablation hook (P9 battery / P3 null proof): AXIOM_NO_SPEC ignores the LLM
        # proposal entirely (static prior + grid candidates), so a run WITH vs WITHOUT
        # the spec isolates the proposal's effect — connected-but-null, not dropped.
        spec_applied = bool(spec_ok and not os.environ.get("AXIOM_NO_SPEC"))
        if not spec_applied:
            spec_prior, spec_cands = None, None

        # ── LLM synthesis
        user_msg = (
            f"ENKF STATE: T={es0['T']}K (±{es0['std_T']}K 1σ), "
            f"Ca={es0['Ca']} (±{es0['std_Ca']}), "
            f"UAf_est={es0['UAf']:.3f}±{es0['std_UA']:.3f}, "
            f"Ca0f_est={es0['Ca0f']:.3f}, "
            f"ensemble_spread_T={es0['ensemble_spread_T']}K.\n\n"
            f"BAYESIAN BELIEF STATE:\n"
            f"  MAP: {top3[0][0]} | P={top3[0][1]:.3f} | {top3[0][2]}\n"
            f"  2nd: {top3[1][0]} | P={top3[1][1]:.3f} | {top3[1][2]}\n"
            f"  3rd: {top3[2][0]} | P={top3[2][1]:.3f} | {top3[2][2]}\n"
            f"  KL-divergence from prior: {b0['kl_div']:.4f} "
            f"(threshold={belief.kl_threshold})\n"
            f"  Belief entropy: {b0['entropy']:.4f}\n\n"
            f"CAUSAL ATTRIBUTION SCORES:\n"
            f"  {fp['attribution']}\n"
            f"  Primary cause: {fp['primary_cause']} "
            f"(fault strength={fp['fault_strength']:.3f})\n\n"
            f"IT-OPTIMAL ACTION:\n"
            f"  Tc_selected={it_act['Tc_selected']}K, "
            f"EIG={it_act['eig']:.5f}, "
            f"safety_critical={safety_crit}\n\n"
            f"ANOMALY: events={anomaly['events']}, "
            f"severity={anomaly['severity']}, "
            f"margin_to_runaway={anomaly['margin']}K, "
            f"risk_index={anomaly['risk_index']:.4f}\n\n"
            f"RAG-RETRIEVED SOP CONTEXT:\n{sop_ctx}"
        )

        llm_t0   = time.time()
        # P5: the explanation must describe the VERIFIED action (it_act's Tc) so the
        # released narrative is governed by explanation-consistency, not free to
        # misstate what was done.
        synthesis, model_used = self._synthesize(user_msg, sops,
                                                 verified_tc=float(it_act["Tc_selected"]))
        llm_time = time.time() - llm_t0

        # ── Closed-loop trajectory simulation — all 4 comparators
        print("  [SIM] Running closed-loop trajectories (AXIOM + 4 baselines)...")
        cfe = CounterfactualEngine(self.cstr)

        # AXIOM — use NAMED pillar objects so the closed-loop belief/EnKF
        # evolution (not just the t=0 snapshot) is retained, logged, and
        # reported. ts_log accumulates the per-step closed-loop time series.
        axiom_belief = HypothesisBelief(self.cstr, kl_threshold=0.18, prior=spec_prior)
        axiom_enkf   = EnsembleKalmanFilter(self.cstr, N=80, rng=rng)
        axiom_enkf.ensemble = axiom_enkf._init_ensemble([y0[0], y0[1], UAf, Ca0f])
        axiom_it     = ConservativeSafetyScreenedController(self.cstr, n_candidates=25)
        ts_log       = []
        axiom_fn = self.axiom_ctrl_fn(Ca0f, UAf, q_frac,
                                      axiom_belief, axiom_it, axiom_enkf,
                                      ts_log=ts_log, rng=rng,
                                      extra_candidates=spec_cands)
        t_a,Ca_a,T_a,Tc_a,_ = self.cstr.simulate_cl(
            t_sim, y0, axiom_fn, Ca0f, UAf, noise, dt, rng=rng, uaf_fn=uaf_fn)

        # Assemble the per-scenario closed-loop time series (real model output).
        if ts_log:
            ts = {
                "t":         np.array([r["t"]         for r in ts_log]),
                "posterior": np.array([r["posterior"] for r in ts_log]),  # (S,10)
                "enkf_mean": np.array([r["enkf_mean"] for r in ts_log]),  # (S,4)
                "enkf_std":  np.array([r["enkf_std"]  for r in ts_log]),  # (S,4)
                "Tc":        np.array([r["Tc"]        for r in ts_log]),
                "eig":       np.array([r["eig"]       for r in ts_log]),
            }
        else:
            ts = None

        # Grid-search controller — ORACLE variant (true fault params; not BO)
        t_b,Ca_b,T_b,Tc_b,_ = cfe.grid_search_agent(t_sim,y0,Ca0f,UAf,noise,dt,rng=rng,uaf_fn=uaf_fn)
        # Grid-search controller — FAIR variant (item 3): same online EnKF
        # estimates AXIOM uses (same init as axiom_enkf), only the action rule
        # differs. Removes the asymmetric-information advantage of the oracle.
        gs_fair_enkf = EnsembleKalmanFilter(self.cstr, N=80, rng=rng)
        gs_fair_enkf.ensemble = gs_fair_enkf._init_ensemble([y0[0], y0[1], UAf, Ca0f])
        t_f,Ca_f,T_f,Tc_f,_ = cfe.grid_search_agent(
            t_sim, y0, Ca0f, UAf, noise, dt, rng=rng, enkf=gs_fair_enkf, uaf_fn=uaf_fn)
        # SIS
        t_s,Ca_s,T_s,Tc_s,_ = cfe.sis_agent(t_sim,y0,Ca0f,UAf,noise,rng=rng,uaf_fn=uaf_fn)
        # PID
        t_p,Ca_p,T_p,Tc_p,_ = cfe.pid_agent(t_sim,y0,Ca0f,UAf,noise,rng=rng,uaf_fn=uaf_fn)
        # Open-loop (counterfactual)
        t_ol,Ca_ol,T_ol,_,_ = cfe.no_intervention(t_sim,y0,Ca0f,UAf,noise,rng=rng,uaf_fn=uaf_fn)

        # ── Metrics
        T_sp = self.cstr.T_sp
        def peak(T):   return round(float(np.max(T)),2)
        def ovsh(T):   return round(float(max(0,np.max(T)-T_sp)),2)
        def ss_err(T): return round(float(abs(T[-1]-T_sp)),2)
        def rec(ts,T): return round(float(next(
            (ts[i] for i in range(len(T)) if T[i]<self.cstr.T_alarm),t_sim)),2)
        def rri_max(T,Ca,Tc,UAf=UAf):
            """Max bounded runaway risk index over the trajectory (item 9)."""
            rri, _ = runaway_risk_index(self.cstr, T, Ca, Tc, UAf)
            return round(float(np.max(rri)),3)

        # Monte Carlo uncertainty bands for AXIOM (n_mc noise realizations),
        # all drawn from the same per-scenario Generator (Phase 4.1).
        peak_samples = []
        for _ in range(n_mc):
            enkf_mc = EnsembleKalmanFilter(self.cstr, N=40, rng=rng)
            belief_mc = HypothesisBelief(self.cstr, prior=spec_prior)
            it_mc = ConservativeSafetyScreenedController(self.cstr, n_candidates=15)
            fn_mc = self.axiom_ctrl_fn(Ca0f,UAf,q_frac,belief_mc,it_mc,enkf_mc,
                                       rng=rng, extra_candidates=spec_cands)
            _,_,T_mc,_,_ = self.cstr.simulate_cl(
                t_sim, y0, fn_mc, Ca0f, UAf, noise+0.5, dt, rng=rng, uaf_fn=uaf_fn)
            peak_samples.append(float(np.max(T_mc)))
        mc_peak_mean = round(float(np.mean(peak_samples)),2)
        mc_peak_std  = round(float(np.std(peak_samples)),2)
        mc_peak_95   = round(float(np.percentile(peak_samples,95)),2)

        # ── Closed-loop belief summary (Phase 2.4): report belief dynamics from
        # the trajectory, NOT the single t=0 update. The LLM was consulted at
        # t=0, so the initial MAP is also retained separately (*_t0).
        if ts is not None and len(ts["posterior"]):
            post_mean   = ts["posterior"].mean(axis=0)        # episode-mean posterior
            cl_map_idx  = int(np.argmax(post_mean))
            cl_map_hyp  = axiom_belief.hypotheses[cl_map_idx]
            cl_map_prob = float(post_mean[cl_map_idx])
            cl_ent      = float(np.mean([float(kl_entropy(p + 1e-12))
                                          for p in ts["posterior"]]))
        else:
            cl_map_hyp, cl_map_prob, cl_ent = top3[0][0], top3[0][1], b0["entropy"]
        cl_kl   = (float(np.mean(axiom_belief.kl_hist))
                   if axiom_belief.kl_hist else 0.0)
        cl_nrev = len(axiom_belief.revision_events)

        # EnKF calibration over the closed loop (Phase 2.6): fraction of the
        # realized T within the filter's 95% CI. The true T seen at each control
        # step aligns with axiom_enkf.history from the start of the run.
        enkf_calib = round(axiom_enkf.compute_calibration(list(T_a)), 3)

        # ── Fail-closed admissibility certificate (Phase 1.3) ───────────────
        # The closed loop executes the verified IT-selected action; the
        # certificate decides whether the LLM recommendation may be presented
        # as authoritative. If any check fails, the system fails closed: it
        # commits the verified safe action and flags the LLM narrative.
        T_run_lim       = self.cstr.T_run
        peak_T_verified = float(np.max(T_a))
        exec_tc         = float(it_act["Tc_selected"])
        rec_tc          = parse_recommended_tc(synthesis)

        # recommendation_safe (item 1): simulate the LLM-recommended Tc from the
        # initial state under the EnKF WORST-CASE plant over the selector horizon
        # and require its PEAK to clear T_run by SAFE_MARGIN. If the setpoint
        # cannot be parsed, treat the recommendation as unsafe (fail closed).
        # n_eval=200: the default 2-point output (endpoints only) MISSES the fast
        # initial-burn transient of the runaway-capable model, under-reporting the
        # peak and falsely blessing an unsafe recommendation (e.g. at S06 the
        # endpoint max is 466 K while the true peak is 493 K). Dense sampling makes
        # this safety gate sound (matches a max_step=0.05 reference to <1 K).
        if rec_tc is not None:
            sol_rec  = self.cstr.simulate((0, it_sel.horizon), y0, Tc=rec_tc,
                                          UAf=uaf_s0, Ca0f=ca0f_s0, n_eval=200)
            peak_rec = float(np.max(sol_rec.y[1]))
            rec_safe = bool(T_run_lim - peak_rec >= SAFE_MARGIN)
        else:
            peak_rec = float("nan")
            rec_safe = False

        # ── Robustness (P6): does the COMMITTED action keep peak T below T_run across
        # an EPISODE-LEVEL Monte-Carlo ensemble drawn from the CALIBRATED parameter
        # uncertainty? DISTINCT from the per-step worst-case screen (P2.5): the screen
        # is the current-moment escalation check; this is the whole-trajectory
        # uncertainty check. Draws: UAf/Ca0f from the EnKF posterior (their calibrated
        # spread); k0 from K0_REL_SIGMA (documented kinetic uncertainty, NOT
        # EnKF-calibrated, author-confirmable). The committed action exec_tc (constant)
        # is applied on each drawn plant; fraction_safe = fraction whose peak clears
        # T_run; robustness holds iff fraction_safe >= ROBUSTNESS_THRESHOLD.
        uaf_mu, uaf_sd = es0["UAf"], max(es0["std_UA"], 1e-3)
        ca0f_mu        = es0["Ca0f"]
        ca0f_sd        = max(float(enkf.ensemble.std(axis=0)[3]), 1e-3)
        rob_horizon    = min(float(t_sim), 12.0)        # peak occurs early; sample densely
        rob_flags      = []
        for _ in range(N_ROBUST):
            uaf_d  = float(np.clip(rng.normal(uaf_mu,  uaf_sd),  0.1, 1.5))
            ca0f_d = float(np.clip(rng.normal(ca0f_mu, ca0f_sd), 0.1, 2.0))
            k0_d   = self.cstr.k0 * float(max(0.1, 1.0 + rng.normal(0.0, K0_REL_SIGMA)))
            cm = CSTRModel(); cm.k0 = k0_d           # perturbed kinetics (k0 not EnKF-calibrated)
            sol_r = cm.simulate((0, rob_horizon), y0, Tc=exec_tc, UAf=uaf_d,
                                Ca0f=ca0f_d, q_frac=q_frac, n_eval=400)
            rob_flags.append(float(np.max(sol_r.y[1])) < T_run_lim)
        fraction_safe = round(float(np.mean(rob_flags)), 3) if rob_flags else 0.0
        robust_ok     = bool(fraction_safe >= ROBUSTNESS_THRESHOLD)
        # ── Structural robustness (HARDENED gate; now ON BY DEFAULT, OPT-OUT via
        # AXIOM_NO_HARDEN_STRUCTURAL=1). ON -> the committed action must ALSO clear T_run by
        # SAFE_MARGIN on a family of two-CSTR-in-series shapes, so wrong-model-SHAPE (structural)
        # mismatch can no longer be silently endorsed (ROBUSTNESS_PLAN.md, Axis 2/3). OPT OUT
        # (AXIOM_NO_HARDEN_STRUCTURAL=1) -> this block is skipped and the certificate is
        # BYTE-IDENTICAL to the locked cs1-live-final study.
        struct_info = None
        if HARDEN_STRUCTURAL:
            _sp = [_two_cstr_peak(self.cstr, _f, y0, exec_tc, ca0f_mu, uaf_mu, q_frac, rob_horizon)
                   for _f in STRUCT_SPLITS]
            struct_ok = all(p <= T_run_lim - SAFE_MARGIN for p in _sp)
            robust_ok = bool(robust_ok and struct_ok)
            struct_info = {"structural_robust": struct_ok,
                           "structural_splits": list(STRUCT_SPLITS),
                           "structural_peaks": [round(p, 1) for p in _sp]}

        # Evidence coverage VERIFIED AGAINST RETRIEVAL (P5): every SOP id cited in the
        # explanation must be in the retrieved set — a hallucinated or stale citation
        # (cited but not retrieved) fails closed. Requires >=1 citation.
        cited_ids     = set(re.findall(r"(?:SOP|REF)-\d+", synthesis or ""))
        retrieved_ids = {r["id"] for r in sops}
        # Decision-critical EVIDENCE COVERAGE (P5b; CS1 analog of CS2 e_min=1.0): when a
        # decision-critical hard-limit SOP (the documented 475 K runaway limit) is retrieved,
        # the released proposal must GROUND in at least one of them. Vacuously satisfied only
        # if none was retrievable, exactly as CS2's coverage is vacuous when no critical
        # constraint has a current corpus entry.
        crit_retrieved    = retrieved_ids & set(DECISION_CRITICAL_SOPS)
        evidence_coverage = (1.0 if not crit_retrieved
                             else (1.0 if (cited_ids & crit_retrieved) else 0.0))
        evidence_ok   = (bool(cited_ids) and cited_ids.issubset(retrieved_ids)
                         and evidence_coverage >= EVIDENCE_E_MIN)
        # Explanation-consistency (P5): the released explanation must STATE the
        # verified committed action. rec_tc is parsed from the explanation TEXT, so a
        # narrative that MISSTATES the action fails — a genuine parse-and-compare, not
        # tautological (the mock states exec_tc, but the check would catch a lie).
        explanation_ok = bool(rec_tc is not None and abs(rec_tc - exec_tc) <= TC_TOL)

        cert = AdmissibilityCertificate(
            converged = bool(np.all(np.isfinite(T_a)) and np.all(np.isfinite(Ca_a))),
            safe      = bool(T_run_lim - peak_T_verified >= SAFE_MARGIN),
            explanation_consistent = explanation_ok,
            evidence_cited    = evidence_ok,
            robustness = robust_ok,    # P6: action safe across the calibrated MC ensemble
            diagnosis_confident = bool(enkf_calib >= DIAGNOSIS_CALIB_THRESHOLD),  # P7
            recommendation_safe = rec_safe,
            margins   = {"T_run_margin":     round(T_run_lim - peak_T_verified, 2),
                         "rec_T_run_margin": (round(T_run_lim - peak_rec, 2)
                                              if rec_tc is not None else None),
                         "safe_margin_req":  SAFE_MARGIN,
                         "robustness_fraction_safe": fraction_safe,
                         "robustness_threshold":     ROBUSTNESS_THRESHOLD,
                         "evidence_coverage":        evidence_coverage,
                         "evidence_e_min":           EVIDENCE_E_MIN,
                         "enkf_calibration":         enkf_calib,
                         "diagnosis_calib_threshold": DIAGNOSIS_CALIB_THRESHOLD},
            recommended_tc = rec_tc, executed_tc = exec_tc)
        if struct_info is not None:                  # record structural verdict only when ON
            cert.margins.update(struct_info)
        if passes(cert):
            cert.reason   = ("admissible: executed action is verified-safe and robust, "
                             "the explanation truthfully states it, and every cited SOP "
                             "was retrieved")
            decision_mode = "llm_endorsed"
        else:
            failed = [n for n, v in (("converged", cert.converged),
                                     ("safe", cert.safe),
                                     ("robustness", cert.robustness),
                                     ("diagnosis_confident", cert.diagnosis_confident),
                                     ("explanation_consistent", cert.explanation_consistent),
                                     ("evidence_cited", cert.evidence_cited),
                                     ("recommendation_safe", cert.recommendation_safe))
                      if not v]
            cert.reason   = (f"FAIL-CLOSED [{', '.join(failed)}]: LLM Tc={rec_tc} "
                             f"(rec peak margin={cert.margins['rec_T_run_margin']}K) "
                             f"vs verified Tc={exec_tc} K; committing verified safe "
                             f"action, LLM narrative not authoritative")
            decision_mode = "fail_closed_override"
        committed_tc = exec_tc   # verified safe action in both modes
        print(f"  [CERT] passes={passes(cert)}  mode={decision_mode}  "
              f"recTc={rec_tc} execTc={exec_tc}  "
              f"margin={cert.margins['T_run_margin']}K rec_safe={rec_safe}")
        if decision_mode == "fail_closed_override":
            print(f"        {cert.reason}")

        m = {
            "scenario_id": sid, "scenario_name": sname,
            "T_initial": y0[1], "Ca_initial": y0[0],
            "Ca0f": Ca0f, "UAf": UAf, "q_frac": q_frac,
            "fault_type": scenario["fault_type"],
            "severity": anomaly["severity"],
            "events": anomaly["events"],
            "margin": anomaly["margin"],
            "risk_index": anomaly["risk_index"],
            # EnKF
            "enkf_T_est":  es0["T"],  "enkf_T_std":  es0["std_T"],
            "enkf_UA_est": es0["UAf"],"enkf_UA_std": es0["std_UA"],
            "enkf_spread_T": es0["ensemble_spread_T"],
            "enkf_calibration": enkf_calib,
            # Belief (closed-loop dynamics; *_t0 = initial assessment shown to LLM)
            "map_hyp":     cl_map_hyp,         "map_prob":    round(cl_map_prob, 4),
            "map_hyp_t0":  top3[0][0],         "map_prob_t0": top3[0][1],
            "belief_kl":   round(cl_kl, 4),    "belief_ent":  round(cl_ent, 4),
            "belief_kl_t0":b0["kl_div"],
            "n_revisions": cl_nrev,
            # Attribution (all five components stored; no hardcoded zeros)
            "attr_primary": fp["primary_cause"],
            "attr_UAf":     fp["attribution"].get("UAf",0),
            "attr_Ca0f":    fp["attribution"].get("Ca0f",0),
            "attr_q_frac":  fp["attribution"].get("q_frac",0),
            "attr_T0":      fp["attribution"].get("T0",0),
            "attr_Tc":      fp["attribution"].get("Tc",0),
            "fault_strength":fp["fault_strength"],
            # protective action
            "Tc_it":     it_act["Tc_selected"],
            "eig":       it_act["eig"],
            "eig_curve": it_act["eig_curve"],
            # RAG
            "top1_sop": sops[0]["id"] if sops else "N/A",
            "top1_score":sops[0]["score"] if sops else 0,
            # LLM
            "llm_model": model_used,
            "llm_time":  round(llm_time,2),
            "total_time":round(time.time()-t_total,2),
            "synthesis": synthesis,
            # P3: the LLM decision-spec (grounded PROPOSAL, no authority) — logged for
            # auditability. spec_applied=False means rejected or ablated (AXIOM_NO_SPEC).
            "decision_spec":  decision_spec,
            "spec_accepted":  bool(spec_ok),
            "spec_applied":   bool(spec_applied),
            "grounded_prior_init": (axiom_belief.prior_hist[0].tolist()
                                    if spec_applied else None),
            # Fail-closed decision (Phase 1.3)
            "certificate":   asdict(cert),
            "cert_passes":   passes(cert),
            "decision_mode": decision_mode,
            "committed_tc":  committed_tc,
            # Trajectories + closed-loop time series
            "traj":{"axiom":(t_a,Ca_a,T_a,Tc_a),
                    "gs_oracle": (t_b,Ca_b,T_b,Tc_b),
                    "gs_fair":   (t_f,Ca_f,T_f,Tc_f),
                    "sis":  (t_s,Ca_s,T_s,Tc_s),
                    "pid":  (t_p,Ca_p,T_p,Tc_p),
                    "ol":   (t_ol,Ca_ol,T_ol,t_ol*0+self.cstr.Tc_nom)},
            "ts":   ts,
            # Performance metrics (gs_oracle = true params; gs_fair = EnKF est.)
            "peak_axiom": peak(T_a), "peak_gs_oracle":  peak(T_b),
            "peak_gs_fair": peak(T_f),
            "peak_sis":   peak(T_s), "peak_pid": peak(T_p),
            "peak_ol":    peak(T_ol),
            "ovsh_axiom": ovsh(T_a), "ovsh_gs_oracle":  ovsh(T_b),
            "ovsh_gs_fair": ovsh(T_f),
            "ovsh_sis":   ovsh(T_s), "ovsh_pid": ovsh(T_p),
            "ss_axiom":   ss_err(T_a),"ss_gs_oracle":   ss_err(T_b),
            "ss_gs_fair": ss_err(T_f),
            "ss_sis":     ss_err(T_s),"ss_pid":  ss_err(T_p),
            "rec_axiom":  rec(t_a,T_a),"rec_gs_oracle":  rec(t_b,T_b),
            "rec_gs_fair": rec(t_f,T_f),
            "rec_sis":    rec(t_s,T_s),"rec_pid": rec(t_p,T_p),
            "rri_axiom":  rri_max(T_a,Ca_a,Tc_a),
            "rri_gs_oracle":     rri_max(T_b,Ca_b,Tc_b),
            "rri_gs_fair":       rri_max(T_f,Ca_f,Tc_f),
            "rri_removal_limited_axiom":
                bool(runaway_risk_index(self.cstr,T_a,Ca_a,Tc_a,UAf)[1].any()),
            # Monte Carlo
            "mc_peak_mean":mc_peak_mean,"mc_peak_std":mc_peak_std,
            "mc_peak_95":  mc_peak_95,
        }

        print(f"  Peak T — AXIOM:{m['peak_axiom']}K  GS-oracle:{m['peak_gs_oracle']}K "
              f"GS-fair:{m['peak_gs_fair']}K  SIS:{m['peak_sis']}K  PID:{m['peak_pid']}K")
        print(f"  Recovery — AXIOM:{m['rec_axiom']}min  "
              f"GS-oracle:{m['rec_gs_oracle']}min  SIS:{m['rec_sis']}min")
        print(f"  Max RRI — AXIOM:{m['rri_axiom']}  GS-oracle:{m['rri_gs_oracle']} "
              f"GS-fair:{m['rri_gs_fair']}")
        print(f"  MC peak: {mc_peak_mean}±{mc_peak_std}K (95th:{mc_peak_95}K)")
        print(f"  LLM synthesis: {llm_time:.1f}s")
        self.results.append(m)
        return m


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION SUITE (12 figures)
# ══════════════════════════════════════════════════════════════════════════════

def savefig(fig, name, d):
    path = f"{d}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {name}.png")


def F1_phase_portraits(cstr, scenarios, results, d):
    fig,axes = plt.subplots(2,5,figsize=(20,8))
    axes = axes.flatten()
    lss  = cstr.lower_stable_ss()
    Ca_ss,T_ss = lss["Ca"], lss["T"]
    for i,(sc,m) in enumerate(zip(scenarios,results)):
        ax = axes[i]
        ax.axhspan(0,    cstr.T_alarm, alpha=0.04, color=C["safe"])
        ax.axhspan(cstr.T_alarm, cstr.T_run, alpha=0.07, color=C["warning"])
        ax.axhspan(cstr.T_run,   510,         alpha=0.09, color=C["critical"])
        t_a,Ca_a,T_a,_ = m["traj"]["axiom"]
        t_b,Ca_b,T_b,_ = m["traj"]["gs_oracle"]
        t_s,Ca_s,T_s,_ = m["traj"]["sis"]
        t_p,Ca_p,T_p,_ = m["traj"]["pid"]
        ax.plot(Ca_b,T_b,color=C["gs_oracle"],   lw=1.3,ls="--",alpha=0.7,label="Grid (oracle)")
        ax.plot(Ca_s,T_s,color=C["sis"],  lw=1.3,ls=":",  alpha=0.7,label="SIS")
        ax.plot(Ca_p,T_p,color=C["pid"],  lw=1.3,ls="-.", alpha=0.7,label="PID")
        ax.plot(Ca_a,T_a,color=C["axiom"],lw=2.0, alpha=0.95,label="AXIOM")
        ax.plot(Ca_a[0],T_a[0],"o",color=SC_COLORS[i],ms=7,zorder=5)
        ax.plot(Ca_ss,T_ss,"*",color=C["safe"],ms=9,zorder=6)
        ax.axhline(cstr.T_alarm,color=C["warning"],ls=":",lw=0.9,alpha=0.6)
        ax.axhline(cstr.T_run,  color=C["critical"],ls=":",lw=0.9,alpha=0.6)
        ax.set_title(f"{sc['id']}",fontsize=9,fontweight="bold",
                     color=SC_COLORS[i])
        ax.set_xlabel("Ca (mol/L)",fontsize=8)
        ax.set_ylabel("T (K)",fontsize=8)
        ax.set_xlim(-0.02,1.15); ax.set_ylim(310,510)
        if i==0: ax.legend(fontsize=6,loc="upper right")
    fig.suptitle("Fig. 1 — Phase Portraits: State Trajectories Across All 10 Scenarios",
                 fontsize=13,fontweight="bold")
    plt.tight_layout()
    savefig(fig,"Fig01_phase_portraits",d)


def F2_timeseries_grid(cstr, scenarios, results, d):
    fig,axes = plt.subplots(10,2,figsize=(14,40))
    for i,(sc,m) in enumerate(zip(scenarios,results)):
        col = SC_COLORS[i]
        t_a,Ca_a,T_a,Tc_a = m["traj"]["axiom"]
        t_b,_,   T_b,Tc_b = m["traj"]["gs_oracle"]
        t_s,_,   T_s,Tc_s = m["traj"]["sis"]
        t_p,_,   T_p,Tc_p = m["traj"]["pid"]
        ax  = axes[i,0]
        ax.fill_between(t_a,cstr.T_alarm,cstr.T_run,alpha=0.06,color=C["warning"])
        ax.fill_between(t_a,cstr.T_run,  510,        alpha=0.08,color=C["critical"])
        ax.plot(t_b,T_b,color=C["gs_oracle"], lw=1.2,ls="--",alpha=0.7,label="Grid (oracle)")
        ax.plot(t_s,T_s,color=C["sis"],lw=1.2,ls=":", alpha=0.7,label="SIS")
        ax.plot(t_p,T_p,color=C["pid"],lw=1.2,ls="-.",alpha=0.7,label="PID")
        ax.plot(t_a,T_a,color=C["axiom"],lw=1.8,label="AXIOM")
        ax.axhline(cstr.T_sp,   color="gray",         ls=":",lw=0.8)
        ax.axhline(cstr.T_alarm,color=C["warning"],   ls=":",lw=0.8)
        ax.axhline(cstr.T_run,  color=C["critical"],  ls=":",lw=0.8)
        ax.set_ylabel("T (K)",fontsize=9); ax.set_ylim(300,510)
        ax.set_title(f"{sc['id']}: {sc['name']}",fontsize=8,
                     fontweight="bold",color=col)
        if i==0: ax.legend(fontsize=7)
        ax2 = axes[i,1]
        ax2.plot(t_b,Tc_b,color=C["gs_oracle"], lw=1.2,ls="--",alpha=0.7)
        ax2.plot(t_s,Tc_s,color=C["sis"],lw=1.2,ls=":", alpha=0.7)
        ax2.plot(t_p,Tc_p,color=C["pid"],lw=1.2,ls="-.",alpha=0.7)
        ax2.plot(t_a,Tc_a,color=C["axiom"],lw=1.8)
        ax2.axhline(cstr.Tc_nom,color="gray",ls=":",lw=0.8)
        ax2.set_ylabel("Tc (K)",fontsize=9); ax2.set_ylim(240,360)
        ax2.set_title(f"{sc['id']} — Coolant Control",fontsize=8,color=col)
    axes[-1,0].set_xlabel("Time (min)",fontsize=10)
    axes[-1,1].set_xlabel("Time (min)",fontsize=10)
    fig.suptitle("Fig. 2 — T and Tc Time-Series: AXIOM vs Grid Search vs SIS vs PID",
                 fontsize=13,fontweight="bold")
    plt.tight_layout()
    savefig(fig,"Fig02_timeseries",d)


def F3_belief_evolution(results, d):
    """
    Real Bayesian posterior over the 10 fault hypotheses as a stacked area over
    the closed-loop trajectory. P(H|obs) sums to 1 at every step. Data come from
    the persisted per-scenario time series m["ts"] (no fabricated distribution).
    """
    n_show = min(6, len(results))
    fig,axes = plt.subplots(2,3,figsize=(15,8))
    axes = axes.flatten()
    hyp_labels = list(FAULT_HYPOTHESES.keys())
    colors_hyp = plt.cm.tab10(np.linspace(0,1,len(hyp_labels)))

    for i,m in enumerate(results[:n_show]):
        ax = axes[i]
        ts = m.get("ts")
        if ts is None or len(ts.get("t", [])) == 0:
            ax.text(0.5, 0.5, "no time series logged", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color=C["neutral"])
            ax.set_title(m["scenario_id"], fontsize=9, fontweight="bold")
            continue
        t    = ts["t"]
        post = ts["posterior"]                       # (S, 10), rows sum to 1
        ax.stackplot(t, post.T, colors=colors_hyp, alpha=0.85,
                     labels=[h.replace("_", " ") for h in hyp_labels])
        ax.set_xlim(float(t[0]), float(t[-1])); ax.set_ylim(0, 1.0)
        ax.set_xlabel("Time (min)", fontsize=8)
        ax.set_ylabel("P(H | obs)", fontsize=8)
        ax.set_title(f"{m['scenario_id']}: MAP={m['map_hyp']} "
                     f"(mean P={m['map_prob']:.2f})  Rev={m['n_revisions']}",
                     fontsize=8, fontweight="bold", color=SC_COLORS[i])
        if i == 0:
            ax.legend(fontsize=5, loc="upper right", ncol=2)

    fig.suptitle("Fig. 3 — Bayesian Hypothesis Posterior Evolution "
                 "(stacked area; real closed-loop P(H|obs) over time)",
                 fontsize=12,fontweight="bold")
    plt.tight_layout()
    savefig(fig,"Fig03_belief_posterior",d)


def F4_enkf_uncertainty(cstr, scenarios, results, d):
    """
    EnKF state-estimate uncertainty: the TIME-VARYING ensemble mean of T with a
    ±2σ band from the TIME-VARYING ensemble std, both taken from the closed-loop
    time series m["ts"]. Replaces the previous constant-σ band drawn on the
    realized trajectory (which was not the filter estimate).
    """
    # Calibration reporting (round 2, item 5/reporting): the all-scenario mean
    # is primary; the excl-S07 mean is supporting context (S07 is the flow fault
    # where q is structurally unobservable — item 8 — so the filter cannot track
    # it). Both are shown; neither replaces the other.
    _cal   = [m["enkf_calibration"] for m in results
              if m.get("enkf_calibration") is not None]
    _cal_x = [m["enkf_calibration"] for m in results
              if m.get("enkf_calibration") is not None and m["scenario_id"] != "S07"]
    cal_all = float(np.mean(_cal))   if _cal   else float("nan")
    cal_xs7 = float(np.mean(_cal_x)) if _cal_x else float("nan")

    fig,axes = plt.subplots(2,5,figsize=(20,8))
    axes = axes.flatten()
    for i,(sc,m) in enumerate(zip(scenarios,results)):
        ax = axes[i]
        col= SC_COLORS[i]
        ts = m.get("ts")
        if ts is None or len(ts.get("t", [])) == 0:
            ax.text(0.5, 0.5, "no time series logged", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color=C["neutral"])
            ax.set_title(sc["id"], fontsize=8, fontweight="bold", color=col)
            continue
        t      = ts["t"]
        T_mean = ts["enkf_mean"][:, 1]      # T is index 1 of [Ca, T, UAf, Ca0f]
        T_std  = ts["enkf_std"][:, 1]
        ax.fill_between(t, T_mean - 2*T_std, T_mean + 2*T_std,
                        alpha=0.25, color=C["axiom"], label="EnKF ±2σ")
        ax.plot(t, T_mean, color=C["axiom"], lw=1.8, label="EnKF mean T")
        ax.axhline(cstr.T_alarm, color=C["warning"], ls=":", lw=0.9)
        ax.axhline(cstr.T_run,   color=C["critical"],ls=":", lw=0.9)
        lo = float(min((T_mean - 2*T_std).min(), cstr.T_sp - 10))
        hi = float(max((T_mean + 2*T_std).max(), cstr.T_run + 10))
        ax.set_ylim(lo, hi)
        ax.set_xlabel("t (min)",fontsize=8)
        ax.set_ylabel("T (K)",fontsize=8)
        ax.set_title(f"{sc['id']}\nmean σ_T={T_std.mean():.2f}K",
                     fontsize=8,fontweight="bold",color=col)
        if i==0: ax.legend(fontsize=6)
    fig.suptitle("Fig. 4 — EnKF State-Estimate Uncertainty: "
                 "Time-Varying Ensemble Mean of T with ±2σ Band\n"
                 f"mean 95% calibration = {cal_all:.2f} (all 10 scenarios, primary)  |  "
                 f"{cal_xs7:.2f} (excl. S07: flow q structurally unobservable, item 8)",
                 fontsize=12,fontweight="bold")
    plt.tight_layout()
    savefig(fig,"Fig04_enkf_uncertainty",d)


def F5_causal_attribution_heatmap(results, d):
    """
    Causal attribution (local sensitivity) heatmap across all scenarios.

    SCOPE CAVEAT (flow): the EnKF state is [Ca, T, UAf, Ca0f] — it does NOT
    estimate the flow fraction q, and q proved unobservable from (T, Ca) when
    we tried augmenting the state (the estimate drifted away from truth and even
    invented a flow fault on a no-flow scenario; see FIXES.md). The scorer
    therefore assumes nominal flow, so the q column is near zero BY DESIGN and
    must not be read as "flow is never a cause". Attribution is scoped to
    cooling (UA), feed (Ca0), feed-T (T0), and control (Tc).
    """
    attrs   = ["UAf","Ca0f","q_frac","T0","Tc"]
    matrix  = np.array([[m["attr_UAf"], m["attr_Ca0f"],
                         m.get("attr_q_frac", 0.0), m.get("attr_T0", 0.0),
                         m["attr_Tc"]]
                        for m in results])
    sc_ids  = [m["scenario_id"] for m in results]

    fig,ax = plt.subplots(figsize=(10,6))
    cmap   = LinearSegmentedColormap.from_list(
        "attr",["#F5F5F5","#90CAF9","#1565C0"])
    im = ax.imshow(matrix,cmap=cmap,aspect="auto",vmin=0,vmax=0.8)
    ax.set_xticks(range(len(attrs)))
    ax.set_xticklabels(["UA (cooling)","Ca0 (feed)","q (flow)†",
                         "T0 (feed T)","Tc (control)"],fontsize=10)
    ax.set_yticks(range(len(sc_ids)))
    ax.set_yticklabels(sc_ids,fontsize=10)
    plt.colorbar(im,ax=ax,label="Normalized causal attribution [0,1]")
    for i in range(len(results)):
        for j in range(len(attrs)):
            v = matrix[i,j]
            ax.text(j,i,f"{v:.2f}",ha="center",va="center",
                    fontsize=9,fontweight="bold" if v>0.5 else "normal",
                    color="white" if v>0.5 else "black")
    ax.set_xlabel("Process Variable",fontsize=11)
    ax.set_ylabel("Scenario",fontsize=11)
    ax.set_title("Fig. 5 — Causal Attribution Scores: "
                 "Root Cause Quantification via Local Sensitivity Analysis",
                 fontsize=11,fontweight="bold")
    fig.text(0.5, 0.005,
             "† q (flow) is not estimated by the EnKF and is unobservable from "
             "(T, Ca); near-zero values do NOT mean flow is not a cause. "
             "Attribution is scoped to cooling, feed, feed-T, and control.",
             ha="center", fontsize=7, color=C["neutral"])
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    savefig(fig,"Fig05_causal_attribution",d)


def F6_it_eig_curves(results, d):
    """
    Real Expected-Information-Gain curve vs candidate coolant temperature for
    the INITIAL decision step (t=0), as computed by the conservative safety-screened controller.
    The curve is the actual ``eig_curve`` returned by the selector (previously
    dropped before it reached the figure), not a placeholder.
    """
    n_show = min(6,len(results))
    fig,axes = plt.subplots(2,3,figsize=(15,8))
    axes = axes.flatten()
    for i,m in enumerate(results[:n_show]):
        ax  = axes[i]
        col = SC_COLORS[i]
        eig_data = m.get("eig_curve",[])
        if eig_data:
            data = sorted(eig_data, key=lambda x: x[1])   # sort by Tc for plotting
            eigs = [e for e,_ in data]
            tcs  = [tc for _,tc in data]
            ax.plot(tcs,eigs,color=col,lw=1.8,marker="o",ms=2.5)
            ax.fill_between(tcs,min(eigs),eigs,alpha=0.15,color=col)
            ax.axvline(m["Tc_it"],color=C["axiom"],ls="--",lw=1.5,
                       label=f"Selected Tc={m['Tc_it']}K")
        else:
            ax.text(0.5,0.5,"no EIG curve logged",ha="center",va="center",
                    transform=ax.transAxes,fontsize=9,color=C["neutral"])
        ax.set_xlabel("Candidate Tc (K)",fontsize=9)
        ax.set_ylabel("Expected Info Gain",fontsize=9)
        ax.set_title(f"{m['scenario_id']}: max EIG={m['eig']:.4f}\n"
                     f"MAP={m['map_hyp']} (mean P={m['map_prob']:.3f})",
                     fontsize=8,fontweight="bold",color=col)
        if i==0: ax.legend(fontsize=8)
    fig.suptitle("Fig. 6 — Information-Theoretic Action Selection: "
                 "Expected Information Gain vs Candidate Coolant Temperature "
                 "(initial decision, t=0)",
                 fontsize=12,fontweight="bold")
    plt.tight_layout()
    savefig(fig,"Fig06_it_eig_curves",d)


def F7_rri_comparison(cstr, scenarios, results, d):
    """Runaway Risk Index trajectories — all scenarios, AXIOM vs Grid Search."""
    fig,axes = plt.subplots(2,5,figsize=(20,8))
    axes = axes.flatten()
    for i,(sc,m) in enumerate(zip(scenarios,results)):
        ax  = axes[i]
        col = SC_COLORS[i]
        t_a,Ca_a,T_a,Tc_a = m["traj"]["axiom"]
        t_b,Ca_b,T_b,Tc_b = m["traj"]["gs_oracle"]
        # Bounded RRI (item 9): in [0, RRI_MAX], no spurious blow-ups.
        rri_a, rl_a = runaway_risk_index(cstr, T_a, Ca_a, Tc_a, m["UAf"])
        rri_b, _    = runaway_risk_index(cstr, T_b, Ca_b, Tc_b, m["UAf"])
        ax.fill_between(t_a, 1.0, 1.5,     alpha=0.07, color=C["warning"])
        ax.fill_between(t_a, 1.5, RRI_MAX, alpha=0.09, color=C["critical"])
        ax.plot(t_b,rri_b,color=C["gs_oracle"],   lw=1.4,ls="--",label="Grid (oracle)",alpha=0.8)
        ax.plot(t_a,rri_a,color=C["axiom"],lw=1.8,label="AXIOM")
        if np.asarray(rl_a).any():     # flag removal-limited (capped) regime
            ax.plot(np.asarray(t_a)[rl_a], np.asarray(rri_a)[rl_a], "v",
                    color=C["critical"], ms=4, alpha=0.7,
                    label="removal-limited")
        ax.axhline(1.0,color=C["warning"], ls=":",lw=1.0,alpha=0.7)
        ax.axhline(1.5,color=C["critical"],ls=":",lw=1.0,alpha=0.7)
        ax.set_ylim(0, RRI_MAX + 0.3)        # fixed, bounded axis (no artifact)
        ax.set_xlabel("t (min)",fontsize=8); ax.set_ylabel("RRI",fontsize=8)
        ax.set_title(f"{sc['id']}\nRRI_max AXIOM:{m['rri_axiom']} "
                     f"GS:{m['rri_gs_oracle']}",fontsize=8,fontweight="bold",color=col)
        if i==0: ax.legend(fontsize=7)
    fig.suptitle("Fig. 7 — Runaway Risk Index (RRI=heat gen/heat removal): "
                 "AXIOM vs Grid Search. RRI>1.5 = imminent runaway.",
                 fontsize=12,fontweight="bold")
    plt.tight_layout()
    savefig(fig,"Fig07_rri_trajectories",d)


def F8_rag_heatmap(rag, scenarios, results, d):
    """RAG semantic similarity heatmap — 10 scenarios x 25 SOPs."""
    queries = [f"T={m['T_initial']}K Ca={m['Ca_initial']:.3f} "
               f"fault {m['fault_type']} events {' '.join(m['events'])}"
               for m in results]
    mat    = rag.full_matrix(queries)
    sop_ids= [doc["id"] for doc in SOP_LIBRARY]
    sc_ids = [m["scenario_id"] for m in results]

    fig,ax = plt.subplots(figsize=(18,6))
    cmap   = LinearSegmentedColormap.from_list(
        "rag",["#FAFAFA","#BBDEFB","#1565C0"])
    im = ax.imshow(mat,cmap=cmap,aspect="auto",vmin=0,vmax=0.55)
    ax.set_xticks(range(len(sop_ids)))
    ax.set_xticklabels(sop_ids,rotation=55,ha="right",fontsize=7)
    ax.set_yticks(range(len(sc_ids)))
    ax.set_yticklabels(sc_ids,fontsize=9)
    plt.colorbar(im,ax=ax,label="TF-IDF cosine similarity")
    for i in range(len(sc_ids)):
        for j in range(len(sop_ids)):
            v = mat[i,j]
            if v > 0.15:
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",
                        fontsize=5.5,
                        color="white" if v>0.35 else "black")
    ax.set_xlabel("SOP Document ID",fontsize=11)
    ax.set_ylabel("Scenario",fontsize=11)
    ax.set_title("Fig. 8 — RAG Retrieval Heatmap: "
                 "Semantic Similarity Between Scenario Fault Queries and 25 SOP Documents",
                 fontsize=11,fontweight="bold")
    plt.tight_layout()
    savefig(fig,"Fig08_rag_heatmap",d)


def F9_mc_uncertainty_bars(results, d):
    """Monte Carlo peak temperature uncertainty — AXIOM robustness."""
    sc_ids = [m["scenario_id"] for m in results]
    means  = [m["mc_peak_mean"] for m in results]
    stds   = [m["mc_peak_std"]  for m in results]
    p95    = [m["mc_peak_95"]   for m in results]
    peaks_gso = [m["peak_gs_oracle"] for m in results]
    peaks_gsf = [m["peak_gs_fair"]   for m in results]
    peaks_a   = [m["peak_axiom"]     for m in results]
    x = np.arange(len(sc_ids))

    fig,ax = plt.subplots(figsize=(13,5))
    ax.bar(x-0.27, peaks_a, 0.26, color=C["axiom"], alpha=0.85,
           label="AXIOM (deterministic)", zorder=3)
    ax.bar(x,      peaks_gso,0.26, color=C["gs_oracle"], alpha=0.85,
           label="Grid Search — oracle", zorder=3)
    ax.bar(x+0.27, peaks_gsf,0.26, color=C["gs_fair"], alpha=0.85,
           label="Grid Search — fair (EnKF est.)", zorder=3)
    ax.errorbar(x-0.27, means, yerr=[
        [m-lo for m,lo in zip(means,[m-s*1.96 for m,s in zip(means,stds)])],
        [hi-m for m,hi in zip(means,[m+s*1.96 for m,s in zip(means,stds)])]
    ], fmt="none", color="black", capsize=4, lw=1.5, label="MC 95% CI", zorder=4)
    ax.scatter(x-0.27, p95, color="purple", s=60, zorder=5,
               marker="^", label="MC 95th percentile")
    ax.axhline(450, color=C["warning"], ls=":", lw=1.0, alpha=0.7, label="T_alarm")
    ax.axhline(475, color=C["critical"],ls=":", lw=1.0, alpha=0.7, label="T_run")
    ax.set_xticks(x); ax.set_xticklabels(sc_ids,fontsize=9)
    ax.set_ylabel("Peak Temperature (K)",fontsize=11)
    ax.set_title("Fig. 9 — Monte Carlo Robustness: "
                 "AXIOM Peak T Distribution (30 noise realizations) vs Grid Search",
                 fontsize=11,fontweight="bold")
    ax.legend(fontsize=8,ncol=3); ax.set_ylim(300,520)
    plt.tight_layout()
    savefig(fig,"Fig09_mc_uncertainty",d)


def F10_radar_4way(results, d):
    """Radar chart: AXIOM vs Grid Search vs SIS vs PID — 6 metrics."""
    labels = ["Peak T↓","Overshoot↓","SS Error↓",
              "Recovery↓","Belief\nEntropy↑","SOP\nPrecision↑"]
    N  = len(labels)
    agents = ["axiom","gs_oracle","gs_fair","sis","pid"]
    acolors= [C["axiom"],C["gs_oracle"],C["gs_fair"],C["sis"],C["pid"]]
    alabs  = ["AXIOM","Grid (oracle)","Grid (fair)","SIS","PID"]

    def normalize(vals, higher_better=False):
        lo,hi = min(vals),max(vals)
        if hi-lo < 1e-9: return [0.5]*len(vals)
        norm = [(v-lo)/(hi-lo) for v in vals]
        return norm if not higher_better else [1-n for n in norm]

    # Collect metric vectors per agent
    metric_dict = {}
    for ag in agents:
        metric_dict[ag] = {
            "peak":    [m[f"peak_{ag}"]  for m in results],
            "ovsh":    [m[f"ovsh_{ag}"]  for m in results],
            "ss":      [m[f"ss_{ag}"]    for m in results],
            "rec":     [m[f"rec_{ag}"]   for m in results],
            "ent":     [m["belief_ent"]  for m in results]
                        if ag=="axiom" else [0.3]*len(results),
            "sop":     [m["top1_score"]  for m in results]
                        if ag=="axiom" else [0.05]*len(results),
        }

    # Mean across scenarios, then normalize
    means = {}
    for ag in agents:
        means[ag] = [
            np.mean(metric_dict[ag]["peak"]),
            np.mean(metric_dict[ag]["ovsh"]),
            np.mean(metric_dict[ag]["ss"]),
            np.mean(metric_dict[ag]["rec"]),
            np.mean(metric_dict[ag]["ent"]),
            np.mean(metric_dict[ag]["sop"]),
        ]

    # Normalize each metric across agents
    norm_scores = {}
    for k in range(N):
        vals    = [means[ag][k] for ag in agents]
        lo,hi   = min(vals),max(vals)
        rng     = hi-lo if hi-lo>1e-9 else 1.0
        high_b  = k>=4   # entropy and SOP: higher = better
        for j,ag in enumerate(agents):
            if ag not in norm_scores: norm_scores[ag] = []
            n = (vals[j]-lo)/rng
            norm_scores[ag].append(n if high_b else 1-n)

    angles = np.linspace(0,2*np.pi,N,endpoint=False).tolist()
    angles += angles[:1]

    fig,ax = plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))
    for ag,col,lab in zip(agents,acolors,alabs):
        vals = norm_scores[ag] + [norm_scores[ag][0]]
        ax.plot(angles,vals,"o-",lw=2,color=col,label=lab)
        ax.fill(angles,vals,alpha=0.1,color=col)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels,fontsize=10)
    ax.set_ylim(0,1)
    ax.set_yticks([0.25,0.5,0.75,1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"],fontsize=7)
    ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.1),fontsize=10)
    ax.set_title("Fig. 10 — Performance Radar: AXIOM vs Grid Search vs SIS vs PID\n"
                 "(normalized, higher = better on each axis)",
                 fontsize=10,fontweight="bold",pad=20)
    plt.tight_layout()
    savefig(fig,"Fig10_radar_4way",d)


def F11_decision_timeline(results, d):
    """Decision timeline: Gantt-style — recovery, protective action, revision events."""
    fig,ax = plt.subplots(figsize=(14,6))
    y      = np.arange(len(results))
    sev_c  = {"critical":C["critical"],"warning":C["warning"],
               "caution":C["warning"],"normal":C["safe"],"shutdown":C["critical"]}

    for i,m in enumerate(results):
        col = sev_c.get(m["severity"],C["neutral"])
        # Recovery bars
        ax.barh(i,     m["rec_axiom"],color=C["axiom"],height=0.22,alpha=0.85,
                label="AXIOM" if i==0 else "")
        ax.barh(i-0.25,m["rec_gs_oracle"],  color=C["gs_oracle"],   height=0.22,alpha=0.75,
                label="Grid (oracle)"    if i==0 else "")
        ax.barh(i-0.50,m["rec_sis"], color=C["sis"],  height=0.22,alpha=0.75,
                label="SIS"   if i==0 else "")
        ax.barh(i-0.75,m["rec_pid"], color=C["pid"],  height=0.22,alpha=0.75,
                label="PID"   if i==0 else "")
        # protective action marker
        ax.plot(0.2, i, "D",color=col,ms=8,zorder=5,
                label="protective action" if i==0 else "")
        # Annotation
        ax.text(max(m["rec_axiom"],m["rec_gs_oracle"],m["rec_sis"],m["rec_pid"])+0.3,
                i-0.35, f"RRI={m['rri_axiom']}|P={m['map_prob']:.2f}",
                va="center",fontsize=7.5,color="black")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{m['scenario_id']}: {m['scenario_name'][:30]}"
                        for m in results],fontsize=8)
    ax.set_xlabel("Recovery time (min)",fontsize=11)
    ax.set_title("Fig. 11 — Recovery Time Comparison: "
                 "AXIOM vs Grid Search vs SIS vs PID Across All Scenarios",
                 fontsize=11,fontweight="bold")
    ax.legend(fontsize=8,loc="lower right",ncol=2)
    ax.axvline(5,color="gray",ls=":",lw=1.0,alpha=0.5)
    ax.text(5.1,-0.7,"5 min",fontsize=8,color="gray")
    plt.tight_layout()
    savefig(fig,"Fig11_decision_timeline",d)


def F12_composite_summary(cstr, scenarios, results, d):
    """Journal-ready composite summary — 3x4 panel."""
    fig = plt.figure(figsize=(20,14))
    gs  = gridspec.GridSpec(3,4,figure=fig,hspace=0.50,wspace=0.38)
    sc_ids = [m["scenario_id"] for m in results]
    x      = np.arange(len(sc_ids))
    w      = 0.16
    # Summary bar panels include BOTH grid-search variants (item 3); the
    # per-scenario S08 timeseries (panel e) keeps the oracle only to avoid clutter.
    agents = ["axiom","gs_oracle","gs_fair","sis","pid"]
    acolors= [C["axiom"],C["gs_oracle"],C["gs_fair"],C["sis"],C["pid"]]
    agents_ps = ["axiom","gs_oracle","sis","pid"]
    alab   = {"axiom":"AXIOM","gs_oracle":"GS-oracle","gs_fair":"GS-fair",
              "sis":"SIS","pid":"PID"}
    nb     = len(agents)

    # (a) Peak T
    ax_a = fig.add_subplot(gs[0,0])
    for j,(ag,col) in enumerate(zip(agents,acolors)):
        ax_a.bar(x+(j-(nb-1)/2)*w,[m[f"peak_{ag}"] for m in results],
                 w,color=col,alpha=0.85,label=alab[ag])
    ax_a.axhline(cstr.T_alarm,color=C["warning"],ls=":",lw=1.0)
    ax_a.axhline(cstr.T_run,  color=C["critical"],ls=":",lw=1.0)
    ax_a.set_xticks(x); ax_a.set_xticklabels(sc_ids,fontsize=7,rotation=45)
    ax_a.set_ylabel("Peak T (K)",fontsize=9); ax_a.set_title("(a) Peak Temperature")
    ax_a.legend(fontsize=6,ncol=2)

    # (b) Overshoot
    ax_b = fig.add_subplot(gs[0,1])
    for j,(ag,col) in enumerate(zip(agents,acolors)):
        ax_b.bar(x+(j-(nb-1)/2)*w,[m[f"ovsh_{ag}"] for m in results],
                 w,color=col,alpha=0.85)
    ax_b.set_xticks(x); ax_b.set_xticklabels(sc_ids,fontsize=7,rotation=45)
    ax_b.set_ylabel("Overshoot (K)",fontsize=9); ax_b.set_title("(b) Temperature Overshoot")

    # (c) Recovery
    ax_c = fig.add_subplot(gs[0,2])
    for j,(ag,col) in enumerate(zip(agents,acolors)):
        ax_c.bar(x+(j-(nb-1)/2)*w,[m[f"rec_{ag}"] for m in results],
                 w,color=col,alpha=0.85)
    ax_c.set_xticks(x); ax_c.set_xticklabels(sc_ids,fontsize=7,rotation=45)
    ax_c.set_ylabel("Recovery (min)",fontsize=9); ax_c.set_title("(c) Recovery Time")

    # (d) Max RRI comparison (bounded, item 9; both grid-search variants, item 3)
    ax_d = fig.add_subplot(gs[0,3])
    ax_d.bar(x-0.27,[m["rri_axiom"] for m in results],0.26,
             color=C["axiom"],alpha=0.85,label="AXIOM")
    ax_d.bar(x,[m["rri_gs_oracle"] for m in results],0.26,
             color=C["gs_oracle"], alpha=0.85,label="GS-oracle")
    ax_d.bar(x+0.27,[m["rri_gs_fair"] for m in results],0.26,
             color=C["gs_fair"],   alpha=0.85,label="GS-fair")
    ax_d.axhline(1.0,color=C["warning"],ls=":",lw=1.0)
    ax_d.axhline(1.5,color=C["critical"],ls=":",lw=1.0)
    ax_d.set_xticks(x); ax_d.set_xticklabels(sc_ids,fontsize=7,rotation=45)
    ax_d.set_ylabel("Max RRI",fontsize=9); ax_d.set_title("(d) Max Runaway Risk Index")
    ax_d.legend(fontsize=7)

    # (e) S08 compound fault — full time-series (oracle GS only, per-scenario)
    ax_e = fig.add_subplot(gs[1,0:2])
    m8 = results[7]  # S08
    for ag,ls in zip(agents_ps,["solid","--",":","-."]):
        t,_,T,_ = m8["traj"][ag]
        ax_e.plot(t,T,color=C[ag],lw=1.6,ls=ls,label=alab[ag],alpha=0.85)
    ax_e.fill_between(m8["traj"]["axiom"][0],cstr.T_alarm,cstr.T_run,
                      alpha=0.07,color=C["warning"])
    ax_e.fill_between(m8["traj"]["axiom"][0],cstr.T_run,520,
                      alpha=0.09,color=C["critical"])
    ax_e.axhline(cstr.T_sp,color="gray",ls=":",lw=0.8)
    ax_e.set_xlabel("Time (min)"); ax_e.set_ylabel("T (K)")
    ax_e.set_title("(e) S08 Cascading Fault — Temperature Response")
    ax_e.set_ylim(300,520); ax_e.legend(fontsize=8,ncol=4)

    # (f) MAP probability vs EIG scatter
    ax_f = fig.add_subplot(gs[1,2])
    sc_x = [m["eig"]      for m in results]
    sc_y = [m["map_prob"] for m in results]
    sc_c = [m["rri_axiom"] for m in results]
    sc = ax_f.scatter(sc_x,sc_y,c=sc_c,s=100,cmap="RdYlGn_r",
                      vmin=0,vmax=RRI_MAX,zorder=5)
    plt.colorbar(sc,ax=ax_f,label="Max RRI")
    for m in results:
        ax_f.annotate(m["scenario_id"],(m["eig"],m["map_prob"]),
                      xytext=(3,3),textcoords="offset points",fontsize=7)
    ax_f.set_xlabel("Expected Info Gain (EIG)",fontsize=9)
    ax_f.set_ylabel("MAP hypothesis P",fontsize=9)
    ax_f.set_title("(f) EIG vs Belief Certainty\n(color=RRI)",fontsize=9)

    # (g) Belief entropy vs fault strength
    ax_g = fig.add_subplot(gs[1,3])
    ax_g.scatter([m["fault_strength"] for m in results],
                  [m["belief_ent"]     for m in results],
                  c=SC_COLORS[:len(results)],s=90,zorder=5)
    for m in results:
        ax_g.annotate(m["scenario_id"],(m["fault_strength"],m["belief_ent"]),
                      xytext=(3,2),textcoords="offset points",fontsize=7)
    ax_g.set_xlabel("Fault Strength",fontsize=9)
    ax_g.set_ylabel("Belief Entropy (nats)",fontsize=9)
    ax_g.set_title("(g) Belief Entropy vs Fault Strength",fontsize=9)

    # (h) % improvement AXIOM vs best baseline (best of BOTH grid-search
    # variants, SIS, PID — item 3 makes this an honest comparison)
    imp_peak = [100*(min(m["peak_gs_oracle"],m["peak_gs_fair"],m["peak_sis"],m["peak_pid"])-
                     m["peak_axiom"])/
                max(1,min(m["peak_gs_oracle"],m["peak_gs_fair"],m["peak_sis"],m["peak_pid"]))
                for m in results]
    imp_rec  = [100*(min(m["rec_gs_oracle"],m["rec_gs_fair"],m["rec_sis"],m["rec_pid"])-
                     m["rec_axiom"])/
                max(0.1,min(m["rec_gs_oracle"],m["rec_gs_fair"],m["rec_sis"],m["rec_pid"]))
                for m in results]
    ax_h = fig.add_subplot(gs[2,0:2])
    ax_h.bar(x-0.2,imp_peak,0.35,color=C["axiom"],alpha=0.85,
             label="Peak T reduction")
    ax_h.bar(x+0.2,imp_rec, 0.35,color=C["neutral"],alpha=0.85,
             label="Recovery improvement")
    ax_h.axhline(0,color="black",lw=0.8)
    ax_h.set_xticks(x); ax_h.set_xticklabels(sc_ids,fontsize=8)
    ax_h.set_ylabel("AXIOM improvement over best baseline (%)",fontsize=9)
    ax_h.set_title("(h) AXIOM Advantage Over Best Alternative Agent")
    ax_h.legend(fontsize=8)

    # (i) MC uncertainty + deterministic peak
    ax_i = fig.add_subplot(gs[2,2:4])
    ax_i.errorbar(x,[m["mc_peak_mean"] for m in results],
                   yerr=[m["mc_peak_std"]*1.96 for m in results],
                   fmt="o",color=C["axiom"],capsize=5,lw=1.8,ms=7,
                   label="AXIOM MC mean ±1.96σ")
    ax_i.scatter(x,[m["mc_peak_95"] for m in results],
                  color="purple",s=60,marker="^",zorder=5,
                  label="MC 95th percentile")
    ax_i.scatter(x,[m["peak_gs_oracle"] for m in results],
                  color=C["gs_oracle"],s=60,marker="s",zorder=5,label="GS-oracle peak")
    ax_i.scatter(x,[m["peak_gs_fair"] for m in results],
                  color=C["gs_fair"],s=55,marker="D",zorder=5,label="GS-fair peak")
    ax_i.axhline(cstr.T_alarm,color=C["warning"],ls=":",lw=1.0,alpha=0.7)
    ax_i.axhline(cstr.T_run,  color=C["critical"],ls=":",lw=1.0,alpha=0.7)
    ax_i.set_xticks(x); ax_i.set_xticklabels(sc_ids,fontsize=8)
    ax_i.set_ylabel("Peak T (K)",fontsize=9)
    ax_i.set_title("(i) Monte Carlo Robustness Analysis — AXIOM")
    ax_i.legend(fontsize=8,ncol=3); ax_i.set_ylim(300,520)

    fig.suptitle(
        "Fig. 12 — AXIOM v2: Comprehensive Performance Summary\n"
        "Exothermic CSTR Abnormal Event Management — 10 Fault Scenarios, "
        "4 Agents, 5 Novel Pillars",
        fontsize=14,fontweight="bold")
    savefig(fig,"Fig12_composite_summary",d)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS TABLE + JSON EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_results(results, save_dir, verbose=True):
    # Persist the per-scenario closed-loop time series and EIG curve as .npz
    # (numpy arrays don't belong in the JSON); previously these were silently
    # dropped, which is why F3/F4/F6 had no data to plot.
    ts_dir = os.path.join(save_dir, "timeseries")
    os.makedirs(ts_dir, exist_ok=True)
    hyp_labels = np.array(list(FAULT_HYPOTHESES.keys()))
    for m in results:
        sid = m["scenario_id"]
        ts  = m.get("ts")
        if ts is not None:
            np.savez(os.path.join(ts_dir, f"{sid}_timeseries.npz"),
                     t=ts["t"], posterior=ts["posterior"],
                     enkf_mean=ts["enkf_mean"], enkf_std=ts["enkf_std"],
                     Tc=ts["Tc"], eig=ts["eig"], hypotheses=hyp_labels)
        eig_curve = m.get("eig_curve")
        if eig_curve:
            arr = np.array(eig_curve, dtype=float)        # (k, 2): [eig, Tc]
            np.savez(os.path.join(ts_dir, f"{sid}_eig_curve.npz"),
                     eig=arr[:, 0], Tc=arr[:, 1])

    # Keep the certificate (small JSON dict) in metrics; drop heavy/array fields
    # (synthesis is saved separately to llm_syntheses.txt).
    clean = [{k:v for k,v in m.items()
              if k not in ["traj","synthesis","eig_curve","ts"]}
             for m in results]
    with open(f"{save_dir}/metrics_v2.json","w") as f:
        json.dump(clean, f, indent=2)

    # Save syntheses separately
    with open(f"{save_dir}/llm_syntheses.txt","w",encoding="utf-8") as f:
        for m in results:
            f.write(f"\n{'='*70}\n")
            f.write(f"{m['scenario_id']}: {m['scenario_name']}\n")
            f.write(f"{'='*70}\n")
            f.write(m["synthesis"]+"\n")

    if not verbose:
        return

    # Console table (PkGSo/PkGSf = grid-search oracle/fair peak)
    hdr = (f"{'ID':<5}{'Scenario':<30}{'PkA':>6}{'PkGSo':>6}{'PkGSf':>6}"
           f"{'PkS':>6}{'PkP':>6}{'RcA':>5}{'RcGSo':>6}"
           f"{'RRIA':>6}{'RRIGSo':>7}{'RRIGSf':>7}{'MAP':>6}{'EIG':>7}")
    print("\n"+"="*116)
    print("AXIOM v2 RESULTS TABLE  (GS-oracle = true params; GS-fair = EnKF estimates)")
    print("="*116)
    print(hdr)
    print("-"*116)
    for m in results:
        print(f"{m['scenario_id']:<5}{m['scenario_name'][:29]:<30}"
              f"{m['peak_axiom']:>6.1f}{m['peak_gs_oracle']:>6.1f}{m['peak_gs_fair']:>6.1f}"
              f"{m['peak_sis']:>6.1f}{m['peak_pid']:>6.1f}"
              f"{m['rec_axiom']:>5.1f}{m['rec_gs_oracle']:>6.1f}"
              f"{m['rri_axiom']:>6.3f}{m['rri_gs_oracle']:>7.3f}{m['rri_gs_fair']:>7.3f}"
              f"{m['map_prob']:>6.3f}{m['eig']:>7.5f}")
    print("="*116)

    # EnKF calibration summary (reporting): all-scenario mean is primary; the
    # excl-S07 mean is supporting context (S07 = flow fault, q unobservable, item 8).
    _c  = [m["enkf_calibration"] for m in results
           if m.get("enkf_calibration") is not None]
    _cx = [m["enkf_calibration"] for m in results
           if m.get("enkf_calibration") is not None and m["scenario_id"] != "S07"]
    if _c:
        print(f"EnKF 95% calibration (coverage): mean={np.mean(_c):.3f} "
              f"(all {len(_c)} scenarios, PRIMARY)  |  {np.mean(_cx):.3f} "
              f"(excl. S07 flow-unobservable, supporting context — item 8)")

    # LaTeX (peak: AXIOM & GS-oracle & GS-fair & SIS & PID)
    print("\nLaTeX rows:")
    for m in results:
        hyp_short = m["map_hyp"].replace("H","H-")
        print(f"{m['scenario_id']} & {m['scenario_name'][:30]} & "
              f"{m['peak_axiom']:.1f} & {m['peak_gs_oracle']:.1f} & "
              f"{m['peak_gs_fair']:.1f} & "
              f"{m['peak_sis']:.1f} & {m['peak_pid']:.1f} & "
              f"{m['rec_axiom']:.1f} & {m['rri_axiom']:.3f} & "
              f"{hyp_short} & {m['map_prob']:.3f} & {m['eig']:.4f} \\\\")


def write_manifest(save_dir, scenario_ids, model, t_start, t_end,
                   screen_sensitivity=None):
    """Persist a run manifest (Phase 4.2): seed, library versions, scenario
    list, resolved model id, timestamps, gate thresholds, and (item 2) the
    safety-screen percentile sensitivity — for reproducibility/provenance."""
    import platform
    versions = {}
    for modname in ("numpy", "scipy", "sklearn", "matplotlib", "anthropic"):
        try:
            versions[modname] = __import__(modname).__version__
        except Exception:
            versions[modname] = "n/a"
    manifest = {
        "base_seed":        BASE_SEED,
        "scenarios":        scenario_ids,
        "llm_model":        model,
        "mock_llm":         MOCK_LLM,
        "tc_tolerance_K":   TC_TOL,
        "safe_margin_K":    SAFE_MARGIN,
        "rri_max":          RRI_MAX,
        "python":           platform.python_version(),
        "library_versions": versions,
        "t_start":          time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
        "t_end":            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_end)),
        "wall_seconds":     round(t_end - t_start, 1),
    }
    if screen_sensitivity is not None:
        manifest["screen_sensitivity"] = screen_sensitivity
    with open(os.path.join(save_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: run_manifest.json (seed={BASE_SEED}, model={model})")
    return manifest


# ══════════════════════════════════════════════════════════════════════════════
# P9 — EVALUATION BATTERY (results, positive and null; each row labelled with what it
# demonstrates AND what it does not). The committed action is the verified controller's
# regardless of the gate, so the gate does NOT make control safe — the controller does;
# the gate protects the human-facing RECOMMENDATION layer.
# ══════════════════════════════════════════════════════════════════════════════

BATTERY_LOOSENED_LIMIT = 500.0   # a STALE/conflicting SOP that claims the runaway
                                 # limit is 500 K (vs the model-owned T_run = 475 K)

def run_evaluation_battery(agent, scenarios, t_sim=25.0, dt=0.25):
    """Reproduce the settled P9 results in one harness. Returns a structured dict and
    prints a labelled table. Numbers must match the canonical pipeline run."""
    T_run = float(agent.cstr.T_run)
    fam = lambda h: next((k for k in ("normal", "cooling", "feed", "compound",
                                      "sensor", "flow", "runaway", "cascade")
                          if k in h), h)
    # base (gate on, spec on) and the grounded->flat ablation (AXIOM_NO_SPEC)
    os.environ.pop("AXIOM_NO_SPEC", None)
    base = [agent.run_scenario(s, t_sim=t_sim, dt=dt, n_mc=1) for s in scenarios]
    os.environ["AXIOM_NO_SPEC"] = "1"
    try:
        nospec = [agent.run_scenario(s, t_sim=t_sim, dt=dt, n_mc=1) for s in scenarios]
    finally:
        os.environ.pop("AXIOM_NO_SPEC", None)
    bid = {m["scenario_id"]: m for m in base}
    nid = {m["scenario_id"]: m for m in nospec}

    # ── Row 1: gate vs ungated-LLM. The ungated-LLM follows the SOP/threshold rule on
    # the OBSERVED T (its blind spot vs the worst-case screen); the SIS baseline IS that
    # controller, so peak_sis is the proxy. The gate fail-closing is where it adds value.
    gate_catch     = [m["scenario_id"] for m in base if m["decision_mode"] != "llm_endorsed"]
    ungated_unsafe = [m["scenario_id"] for m in base if m["peak_sis"] > T_run]
    s06 = bid.get("S06", {})
    row1 = {
        "gate_catches": gate_catch,
        "ungated_LLM_unsafe_action": ungated_unsafe,
        "S06_unrecoverable_even_verified_action_unsafe":
            bool(s06 and s06["peak_axiom"] > T_run),
        "note": ("control safety is the controller's, not the gate's (committed action "
                 "is the verified controller's regardless of gating). At S06 NO action "
                 "is safe (even max cooling peaks >T_run); the gate's value there is "
                 "refusing to present an unsafe recommendation as fine and abstaining "
                 "with a warning. SOP-threshold action is safe at 8/10, so action-"
                 "protection is concentrated at S06; broader value is S07 (diagnosis) "
                 "and the knowledge-perturbation row."),
    }
    # ── Rows 2/3: nulls (findings 3 and 4)
    action_same = all(abs(bid[s]["committed_tc"] - nid[s]["committed_tc"]) < 1e-9
                      for s in bid)
    map_exact = sum(bid[s]["map_hyp"] == nid[s]["map_hyp"] for s in bid)
    map_fameq = sum(fam(bid[s]["map_hyp"]) == fam(nid[s]["map_hyp"]) for s in bid)
    row23 = {
        "selector_belief_independent_committed_action_unchanged": bool(action_same),
        "grounded_vs_flat_diagnosis_unchanged_hypothesis_exact": "%d/%d" % (map_exact, len(bid)),
        "grounded_vs_flat_diagnosis_unchanged_family": "%d/%d" % (map_fameq, len(bid)),
        "note": "spec ON vs OFF (AXIOM_NO_SPEC): action and diagnosis identical — the "
                "proposal is wired (P3) but consequentially null (findings 3 and 4).",
    }
    # ── Row 4: knowledge-perturbation false-release + the stale-limit invariant.
    gated_fr   = [m["scenario_id"] for m in base
                  if m["peak_axiom"] > T_run and m["decision_mode"] == "llm_endorsed"]
    ungated_fr = [m["scenario_id"] for m in base
                  if T_run < m["peak_axiom"] <= BATTERY_LOOSENED_LIMIT]
    # unretrieved/stale CITATION: a synthesis citing a non-retrieved SOP fails closed
    orig_syn = agent._synthesize
    agent._synthesize = (lambda user_msg, sops, verified_tc=None:
                         (mock_synthesis(["SOP-999"], rec_tc=verified_tc or 285.0), "mock"))
    try:
        m_cite = agent.run_scenario(scenarios[0], t_sim=8.0, dt=dt, n_mc=1)
    finally:
        agent._synthesize = orig_syn
    row4 = {
        "stale_limit_invariant_T_run": T_run,            # model-owned, never the SOP's
        "loosened_limit_claimed_by_stale_SOP": BATTERY_LOOSENED_LIMIT,
        "gated_false_release": gated_fr,                 # [] — safe uses T_run=475
        "ungated_loosened_false_release": ungated_fr,    # [S06] — released under 500 K
        "unretrieved_citation_gated_evidence_cited": bool(m_cite["certificate"]["evidence_cited"]),
        "unretrieved_citation_gated_mode": m_cite["decision_mode"],
        "note": "the controller/certificate use the model-owned T_run (475 K), never "
                "the SOP's number, so a loosened-limit SOP cannot cause a release the "
                "true limit would forbid; and evidence_cited (cited subset of retrieved) "
                "fails closed on an unretrieved citation. Gated zero is a real contrast "
                "to the genuinely nonzero ungated-loosened baseline.",
    }
    battery = {"row1_gate_vs_ungated_LLM": row1, "row2_3_nulls": row23,
               "row4_knowledge_perturbation": row4}
    print("\n" + "═" * 70 + "\n  P9 EVALUATION BATTERY\n" + "═" * 70)
    print("  R1 gate catches (recommendation-layer): %s | ungated-LLM unsafe action: %s"
          % (gate_catch, ungated_unsafe))
    print("     S06 unrecoverable (no safe action): %s — gate WARNS, does not make safe"
          % row1["S06_unrecoverable_even_verified_action_unsafe"])
    print("  R2/3 nulls: action unchanged=%s | diagnosis unchanged exact=%s family=%s"
          % (action_same, row23["grounded_vs_flat_diagnosis_unchanged_hypothesis_exact"],
             row23["grounded_vs_flat_diagnosis_unchanged_family"]))
    print("  R4 stale-limit: T_run invariant=%.0fK | gated false-release=%s vs "
          "ungated-loosened=%s | unretrieved-cite evidence_cited=%s (%s)"
          % (T_run, gated_fr, ungated_fr,
             row4["unretrieved_citation_gated_evidence_cited"], m_cite["decision_mode"]))
    return battery


# ══════════════════════════════════════════════════════════════════════════════
# P10 — LLM VALUE AXES. CRITICAL integrity point: in MOCK the mock states exec_tc and
# cites only retrieved SOPs, so explanation-consistency and citation-grounding are
# 10/10 BY CONSTRUCTION — that is a property of the GATE, not the model. Presenting
# such a number as "LLM performance" would be the most damaging overclaim in the
# paper. So each axis is reported as TWO distinct things: the gate-enforced property
# (mock, construction-true, with a fail-closed proof) and the LLM's RAW rate (the
# substantive number, DEFERRED to the live run). Only auditability is measurable now.
# ══════════════════════════════════════════════════════════════════════════════

def llm_value_axes(results):
    """Return the LLM value axes, each split into the gate-enforced (mock,
    construction-true) property and the LLM's raw rate (deferred to live)."""
    n = len(results)
    expl_ok = sum(1 for m in results if m["certificate"]["explanation_consistent"])
    cite_ok = sum(1 for m in results if m["certificate"]["evidence_cited"])
    audit_fields = ["decision_spec", "certificate", "synthesis", "committed_tc",
                    "decision_mode", "map_hyp", "ts"]
    trace_complete = sum(1 for m in results
                         if all(m.get(k) is not None for k in audit_fields))
    axes = {
        "explanation_consistency": {
            "gate_enforced_rate_mock": "%d/%d" % (expl_ok, n),
            "basis": "GATE-ENFORCED, construction-true in mock (the mock states "
                     "exec_tc); the gate fail-closes when violated "
                     "(test_explanation_consistency_can_fail_on_tampered_text). This is "
                     "a property of the GATE, not LLM performance.",
            "llm_raw_faithfulness_rate": "DEFERRED to live (the substantive value-axis "
                                         "number: how often a real model's stated action "
                                         "matches the verified one before the gate acts)",
        },
        "citation_grounding": {
            "gate_enforced_rate_mock": "%d/%d" % (cite_ok, n),
            "basis": "GATE-ENFORCED (cited subset of retrieved); the gate fail-closes on "
                     "a hallucinated/unretrieved citation (P9 Row 4). A property of the "
                     "GATE, not LLM performance.",
            "llm_raw_hallucination_rate": "DEFERRED to live (the substantive number: a "
                                          "real model's raw citation-hallucination rate)",
        },
        "auditability": {
            "trace_completeness": "%d/%d" % (trace_complete, n),
            "basis": "PRESENT RESULT (model-independent): each decision logs the "
                     "decision_spec, the 7-term certificate + margins, the synthesis, "
                     "committed_tc, decision_mode, and the EnKF/belief time series — a "
                     "complete, replayable audit trace.",
        },
    }
    print("\n" + "═" * 70 + "\n  P10 LLM VALUE AXES\n" + "═" * 70)
    print("  explanation-consistency  GATE-ENFORCED (mock, by construction): %d/%d"
          % (expl_ok, n))
    print("     -> LLM raw faithfulness rate: DEFERRED to live (the substantive number)")
    print("  citation grounding       GATE-ENFORCED (mock, by construction): %d/%d"
          % (cite_ok, n))
    print("     -> LLM raw hallucination rate: DEFERRED to live (the substantive number)")
    print("  auditability             PRESENT RESULT: trace complete %d/%d"
          % (trace_complete, n))
    return axes


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n"+"╔"+"═"*66+"╗")
    print("║  AXIOM v2 — Full Journal Study                                   ║")
    print("║  Five-Pillar Agentic AI for CSTR Safety Management               ║")
    print("╚"+"═"*66+"╝")

    cstr     = CSTRModel()
    rag      = RAGEngine(SOP_LIBRARY)
    registry = MCPToolRegistry()

    # Register the MCP-style tools the agent actually calls. (The "optimize"
    # tool was registered but never invoked in run_scenario — removed in
    # Phase 4.4 so the registry cannot diverge from the executed path.)
    registry.register("detect",
        lambda T,Ca,UAf=1.0,dT_dt=None:
            cstr.detect(T,Ca,UAf,dT_dt),
        "CSTR anomaly detection with risk index")
    registry.register("retrieve_sop",
        lambda query,top_k=4: rag.retrieve(query,top_k),
        "RAG over 25-document SOP knowledge base")
    registry.register("belief_update",
        lambda obs_T,obs_Ca,enkf_state,belief,t=None:
            belief.update(obs_T,obs_Ca,enkf_state,t),
        "Bayesian posterior update over 10 fault hypotheses")
    registry.register("it_action",
        lambda Ca,T,posterior,belief,safety_crit,it_sel,**kw:
            it_sel.select(Ca,T,posterior,belief,safety_crit,**kw),
        "Information-theoretic action selection (max EIG)")
    registry.register("causal_attr",
        lambda Ca,T,enkf_state,Tc,scorer:
            scorer.fingerprint(Ca,T,enkf_state,Tc),
        "Causal attribution via local sensitivity analysis")

    all_scenarios = build_scenarios(cstr)

    # Optional fast-iteration subset, e.g. AXIOM_SCENARIOS="S03,S06,S08".
    # The 12-figure suite assumes all 10 scenarios (some panels index by
    # position), so it is skipped for partial subsets.
    subset = os.environ.get("AXIOM_SCENARIOS", "").strip()
    if subset:
        keep      = {s.strip().upper() for s in subset.split(",") if s.strip()}
        scenarios = [s for s in all_scenarios if s["id"].upper() in keep]
        print(f"  [subset] AXIOM_SCENARIOS -> {[s['id'] for s in scenarios]}")
    else:
        scenarios = all_scenarios
    full_run = (len(scenarios) == len(all_scenarios))

    agent     = AXIOMAgentV2(API_KEY, cstr, rag, registry, mock=MOCK_LLM)

    # Run scenarios, persisting incrementally so a late failure (e.g. an LLM
    # outage on scenario k) does not lose scenarios 1..k-1 (Phase 4.2).
    t_start = time.time()
    results = []
    # AXIOM_NMC overrides the Monte-Carlo band sample count (default = run_scenario's
    # 30); set e.g. AXIOM_NMC=3 for a fast end-to-end pipeline/figure check. The
    # MC bands only widen uncertainty shading — the decision/certificate are
    # unaffected — so this does not change which action is committed.
    _nmc_env = os.environ.get("AXIOM_NMC")
    _nmc_kw  = {"n_mc": int(_nmc_env)} if _nmc_env else {}
    for sc in scenarios:
        m = agent.run_scenario(sc, t_sim=25.0, dt=0.25, **_nmc_kw)
        results.append(m)
        export_results(results, RESULT_DIR, verbose=False)   # checkpoint
    t_end = time.time()

    # Safety-screen percentile sensitivity (item 2): for S08, compare the
    # default ~95% one-sided worst case (5th UAf / 95th Ca0f percentile) with a
    # more conservative ~99% one (1st / 99th), on an ensemble seeded at S08.
    screen_sens = None
    s08 = next((s for s in all_scenarios if s["id"] == "S08"), None)
    if s08 is not None:
        e = EnsembleKalmanFilter(cstr, N=80,
                                 rng=np.random.default_rng(scenario_seed("S08")))
        e.ensemble = e._init_ensemble([s08["y0"][0], s08["y0"][1],
                                       s08["UAf"], s08["Ca0f"]])
        screen_sens = {
            "scenario": "S08",
            "true_UAf": s08["UAf"], "true_Ca0f": s08["Ca0f"],
            "worst_case_5_95":  list(np.round(e.screen_params(5, 95), 4)),
            "worst_case_1_99":  list(np.round(e.screen_params(1, 99), 4)),
            "note": "(UAf_lo, Ca0f_hi); lower UAf / higher Ca0f = more conservative",
        }

    # Final export (with console table + LaTeX) and run manifest.
    export_results(results, RESULT_DIR)
    write_manifest(RESULT_DIR, [s["id"] for s in scenarios],
                   agent.model if not MOCK_LLM else "mock", t_start, t_end,
                   screen_sensitivity=screen_sens)

    # Print tool timing
    print("\nMCP Tool Timing:")
    for name,info in registry.timing().items():
        print(f"  {name}: mean={info['mean']}ms  calls={info['total']}")

    # Generate all 12 figures (full 10-scenario run only)
    if full_run:
        print("\nGenerating 12 journal figures...")
        F1_phase_portraits(cstr,scenarios,results,RESULT_DIR)
        F2_timeseries_grid(cstr,scenarios,results,RESULT_DIR)
        F3_belief_evolution(results,RESULT_DIR)
        F4_enkf_uncertainty(cstr,scenarios,results,RESULT_DIR)
        F5_causal_attribution_heatmap(results,RESULT_DIR)
        F6_it_eig_curves(results,RESULT_DIR)
        F7_rri_comparison(cstr,scenarios,results,RESULT_DIR)
        F8_rag_heatmap(rag,scenarios,results,RESULT_DIR)
        F9_mc_uncertainty_bars(results,RESULT_DIR)
        F10_radar_4way(results,RESULT_DIR)
        F11_decision_timeline(results,RESULT_DIR)
        F12_composite_summary(cstr,scenarios,results,RESULT_DIR)
        print("  Figures: Fig01-Fig12 (300 DPI PNG)")
    else:
        print("\n[subset] Skipping 12-figure suite (requires all 10 scenarios).")

    print(f"\n✓ All outputs saved to: {os.path.abspath(RESULT_DIR)}/")
    print("  Data:    metrics_v2.json + llm_syntheses.txt")


if __name__ == "__main__":
    main()