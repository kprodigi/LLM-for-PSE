# -*- coding: utf-8 -*-
"""
Real MCP server for Case Study 2 (methanol-water distillation) grounding tools,
plus the tool-use proposer that drives them.

Exposes ONLY *advisory grounding* tools to the LLM:
  - retrieve_constraints(query):  CURRENT versioned, provenance-tracked product-purity
                                  / operability specs (sparse constraint channel)
  - retrieve_knowledge(query):    design rationale / heuristics (dense channel)
  - simulate_design(reflux_ratio, DtoF, feed_stage): READ-ONLY evaluation on the trusted
                                  process model (real Aspen Plus V14) -> purities, reboiler
                                  duty, convergence.

The optimizer and the admissibility certificate are NOT exposed -- they remain
harness-owned, so the LLM grounds its spec while the verifier keeps sole authority
over the released design (the framework's safety invariant).

Standalone stdio MCP server (cross-process interoperability):
    python -m case2_flowsheet.mcp_server
"""
import json
import os
import re

from case2_flowsheet.core import mcp_bridge

TOOLUSE_MAX_TOKENS = 8000   # per-call output cap for the TOOL-USE proposer. Raised from 4096
                            # (= one-shot LLM_MAX_TOKENS) before the live CS2 runs: CS1's smoke
                            # showed the model writes ~700 tokens of analysis prose around its
                            # JSON despite the ONLY-JSON instruction, and CS2 specs are much
                            # larger than CS1's -- 4096 risks the same mid-JSON truncation that
                            # contaminated CS1's first probe. Billing is by ACTUAL tokens, so
                            # untruncated turns cost the same. One-shot stays at the proven 4096.


def make_grounding_server(flowsheet, rag, feed_z, name="flowsheet-grounding",
                          expose_retrieval=True):
    """FastMCP server whose tools close over the live design context.

    expose_retrieval=True (default, UNCHANGED): [retrieve_constraints, retrieve_knowledge,
    simulate_design]. expose_retrieval=False (+MCP-only ablation cell): only simulate_design
    is registered (no retrieval tools), isolating MCP's standalone contribution from RAG's."""
    import numpy as np

    def retrieve_constraints(query: str) -> str:
        """Retrieve the CURRENT versioned product-purity / operability constraint specs (with source ids and versions). Cite ONLY ids returned here; never invent a threshold."""
        return json.dumps(rag.retrieve_constraint(query), default=str)

    def retrieve_knowledge(query: str, top_k: int = 4) -> str:
        """Retrieve provenance-tracked design rationale / heuristics passages (dense channel)."""
        hits = rag.retrieve(query, top_k=top_k)
        return json.dumps([{"source_id": h.get("source_id"), "title": h.get("title"),
                            "text": h.get("text"), "version": h.get("version")}
                           for h in hits], default=str)

    def simulate_design(reflux_ratio: float, DtoF: float, feed_stage: int) -> str:
        """Evaluate a candidate distillation design on the trusted process model (real Aspen Plus V14): returns product purities, reboiler duty, convergence, AND a worst-case screen over the feed-composition uncertainty band (whether the design still meets spec if the feed drifts), so the advice reflects the worst plausible feed. READ-ONLY; commits no design."""
        from case2_flowsheet.flowsheet_copilot import constraint_margins
        decision = {"reflux_ratio": float(reflux_ratio), "DtoF": float(DtoF),
                    "feed_stage": int(feed_stage)}
        try:
            res = flowsheet.evaluate(decision, feed_z=feed_z)
        except Exception as exc:
            return json.dumps({"error": "evaluation_failed: %s" % exc})
        scalars = {k: (round(float(v), 5)
                       if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
                       else v)
                   for k, v in res.items()
                   if v is None or isinstance(v, (int, float, bool, str, np.floating))}
        # WORST-CASE SCREEN over the feed-composition uncertainty band -- symmetric with
        # case1_reactor.simulate_setpoint, which screens the worst-case degraded plant. Evaluate
        # the band edges and report the worst feasibility/margin so the proposer sees robustness
        # to feed drift (advisory only; the gate re-verifies the committed design independently).
        try:
            cfg  = getattr(flowsheet, "config", {}) or {}
            cons = cfg.get("constraints", [])
            d    = cfg.get("robustness", {}).get("delta", 0.0)
            z0   = float(feed_z) if feed_z is not None else float(getattr(flowsheet, "z_nom", 0.5))
            m0   = constraint_margins(res, cons)
            worst_m, worst_feas = m0["min_margin"], bool(m0["feasible"])
            for ze in (z0 - d, z0 + d):
                me = constraint_margins(flowsheet.evaluate(decision, feed_z=float(ze)), cons)
                worst_m = min(worst_m, me["min_margin"]); worst_feas = worst_feas and me["feasible"]
            scalars["feed_band"] = [round(z0 - d, 5), round(z0 + d, 5)]
            scalars["worst_case_min_margin"] = round(float(worst_m), 5)
            scalars["worst_case_feasible_over_feed_band"] = bool(worst_feas)
        except Exception:
            pass        # the screen is advisory; the nominal result is always returned
        return json.dumps(scalars, default=str)

    tools = ([retrieve_constraints, retrieve_knowledge, simulate_design] if expose_retrieval
             else [simulate_design])
    return mcp_bridge.make_server(name, tools)


TOOLUSE_PROPOSER_SYSTEM = (
    "You are a process-engineering PROPOSER for an Aspen Plus distillation column. You have "
    "tools: retrieve_constraints(query) for the CURRENT versioned purity/operability specs, "
    "retrieve_knowledge(query) for design rationale, and simulate_design(reflux_ratio, DtoF, "
    "feed_stage) to check a candidate design on the trusted model (it also reports whether the "
    "design still meets spec under feed-composition drift). GROUND your spec: take "
    "thresholds from retrieve_constraints (do NOT invent them; cite their source ids and "
    "versions), and you may probe candidates with simulate_design. You do NOT decide the "
    "final numbers -- an optimizer, the simulator, and an admissibility certificate verify "
    "them downstream. When finished, return ONLY a JSON object with keys 'specs' (a JSON ARRAY "
    "of spec objects, each with 'objective', 'decision_variables', 'constraints' [name, relation, "
    "threshold, source_id, source_version]), 'recommendation' {reflux_ratio, DtoF, feed_stage}, "
    "and 'cited_constraints' (source ids). Constraints: include ONLY the corpus constraints "
    "returned by retrieve_constraints, copied with their source_id and source_version verbatim; "
    "relation must be strictly '>=' or '<='; do NOT add derived or meta-constraints (e.g. "
    "convergence checks) -- the trusted simulator enforces those downstream."
)


TOOLUSE_PROPOSER_SYSTEM_MCP_ONLY = (
    "You are a process-engineering PROPOSER for an Aspen Plus distillation column. You have "
    "ONE tool: simulate_design(reflux_ratio, DtoF, feed_stage) to check a candidate design on "
    "the trusted model. Probe candidates with simulate_design to find a feasible, energy-"
    "efficient design. You have NO constraint-retrieval tool, so do NOT cite sources and "
    "return an empty 'cited_constraints' list. You do NOT decide the final numbers -- an "
    "optimizer, the simulator, and an admissibility certificate verify them downstream. When "
    "finished, return ONLY a JSON object with keys 'specs' (a JSON ARRAY of spec objects, each "
    "with 'objective', 'decision_variables', 'constraints' [name, relation, threshold, "
    "source_id, source_version]), 'recommendation' {reflux_ratio, DtoF, feed_stage}, and "
    "'cited_constraints' (source ids). Constraint relations must be strictly '>=' or '<='; do "
    "NOT add derived or meta-constraints (e.g. convergence checks) -- the trusted simulator "
    "enforces those downstream."
)


def _dv_schema(cfg):
    """The SAME decision-variable schema language the (locked, parse-proven) one-shot
    proposer_prompt uses, built from the SAME config source of truth — the smoke showed the
    model treats an under-specified schema loosely (bare-string dvs, ad-hoc shapes)."""
    dvs = [dv for dv in cfg["decision_vars"] if dv.get("optimize")]
    dv_desc = "; ".join(f"{dv['name']} ({dv['type']}, range {dv['lower']}-{dv['upper']})"
                        for dv in dvs)
    nstages = cfg["system"]["column"]["n_stages"]
    return (f"The column is FIXED at {nstages} stages -- the number of stages is NOT a decision "
            "variable. The ONLY decision variables are EXACTLY these; use these EXACT names, "
            "propose bounds WITHIN these ranges, and do NOT add, rename, or remove any: "
            f"{dv_desc}. Each entry in 'decision_variables' must be an OBJECT "
            "{name,type,lower_bound,upper_bound} -- never a bare name string. "
            "The 'recommendation' MUST use these same keys. "
            # full schema parity with the locked one-shot proposer_prompt (round 5: a single spec
            # citing only 4 sources gave evidence_coverage 0.75 -> the gate correctly failed closed;
            # the one-shot wording below is what kept the locked live run's citations complete):
            "'specs' is a list of 1-3 candidate specifications. Each spec's 'objective' must "
            "contain the literal string 'minimize reboiler_duty'. Each spec also carries "
            "'uncertainty' (list of {parameter,distribution,range}) and 'required_evidence' "
            "(source_ids). 'cited_constraints' must be the source_ids of the decision-critical "
            "constraints you used, from the CURRENT corpus versions.")


def _dv_canonical(cfg):
    return {dv["name"]: {"name": dv["name"], "type": dv["type"],
                         "lower_bound": dv["lower"], "upper_bound": dv["upper"]}
            for dv in cfg["decision_vars"] if dv.get("optimize")}


def tooluse_decision_spec(client, model, flowsheet, rag, feed_z, query=None, mcp_only=False):
    """Tool-use proposer: the LLM grounds its spec by calling the MCP grounding tools,
    then returns a decision specification. Returns (spec_dict, tool_call_log). `client`
    is an Anthropic-style client (real or the scripted mock from mcp_bridge).

    mcp_only=True (+MCP-only ablation cell): expose simulate_design ONLY (no retrieval
    tools) and use the no-citation system prompt, isolating MCP's standalone contribution."""
    server = make_grounding_server(flowsheet, rag, feed_z, expose_retrieval=not mcp_only)
    cfg = getattr(flowsheet, "config", None)
    if cfg is None:
        from case2_flowsheet.flowsheet_copilot import CONFIG as cfg
    system = ((TOOLUSE_PROPOSER_SYSTEM_MCP_ONLY if mcp_only else TOOLUSE_PROPOSER_SYSTEM)
              + "\n" + _dv_schema(cfg))
    if mcp_only:
        user = query or ("Probe candidate designs with simulate_design and return an "
                         "energy-minimizing decision specification.")
    else:
        user = query or ("Compile a machine-checkable decision specification for an "
                         "energy-minimizing design. Ground every threshold via the tools.")
    text, tool_log = mcp_bridge.run_grounded(
        client, model, system, user, server,
        max_tokens=TOOLUSE_MAX_TOKENS)
    # Parse the LAST complete JSON object (extract_last_json): the loop accumulates text across
    # turns, so prose/brace noise can precede the spec and a truncated tail must not break it.
    cand = mcp_bridge.extract_last_json(text)
    try:
        spec = json.loads(cand) if cand else {}
    except Exception:
        spec = {}
    # Shape normalization (observed live, 2026-06-10 smoke): the model sometimes emits "specs"
    # as a single spec OBJECT instead of a 1-element ARRAY -- valid content, wrong shape, and
    # plan() indexes specs[0]. Wrapping it is benign canonicalization (never content invention);
    # anything not spec-shaped is left alone and still fails closed downstream.
    if isinstance(spec, dict) and isinstance(spec.get("specs"), dict):
        if {"objective", "decision_variables", "constraints"} & set(spec["specs"].keys()):
            spec["specs"] = [spec["specs"]]
    # Unknown-source constraint filter (observed live, smoke round 3): the model appended an
    # invented meta-constraint (PROCESS-MODEL-INTEGRITY, relation "==") to an otherwise fully
    # citation-faithful 10-constraint spec, and the all-or-nothing validate_spec rejected the
    # whole spec. plan()'s _normalize_spec_constraints ALREADY drops constraints whose source_id
    # maps to nothing -- but only AFTER validation, so an unverifiable extra can veto verifiable
    # content. Apply the same known-source criterion BEFORE validation, in the tool-use path
    # only. Corpus-sourced constraints still face the full strict checks (a bad relation on a
    # REAL constraint still rejects); the dropped entries remain auditable in the verbatim raw.
    known = {d.get("source_id") for d in getattr(rag, "docs", []) if d.get("source_id")}
    canon = _dv_canonical(cfg)
    if isinstance(spec, dict) and isinstance(spec.get("specs"), list):
        for s in spec["specs"]:
            if isinstance(s, dict) and isinstance(s.get("constraints"), list):
                # Record what the filter removed (A8): when EVERY constraint cites an unknown
                # source (the no-retrieval cells fabricating pseudo-sources like sim_obs_*),
                # plan() must classify the failure as a fail-closed UNGROUNDED abstention (a
                # measured fabrication finding), not capture contamination.
                _dropped = [c.get("source_id") for c in s["constraints"]
                            if isinstance(c, dict) and c.get("source_id") not in known]
                s["constraints"] = [c for c in s["constraints"]
                                    if isinstance(c, dict) and c.get("source_id") in known]
                if _dropped:
                    s["_dropped_unknown_sources"] = _dropped
            # Decision-variable SHAPE canonicalization (observed live, smoke round 4: the model
            # emitted a DICT keyed by variable name with lower/upper key aliases; bare-string
            # lists are another plausible paraphrase). Shape-only: dict-of-name -> list, the
            # lower/upper aliases -> *_bound, missing fields completed from the AUTHOR-owned
            # canonical entry for KNOWN names. The model's own stated bounds are KEPT (the
            # existing post-validation _normalize_spec_decision_vars clamps them to config);
            # unknown names drop (that normalization would drop them anyway); content never
            # invented; strict validation on the result is unchanged.
            # Objective-string canonicalization (observed live, smoke round 6: 'Minimise
            # reboiler duty (kW) ...' -- British spelling + space -- failed the literal
            # 'reboiler_duty' substring check on 3 otherwise-valid specs). Rewrite ONLY when
            # the semantics are regex-verified (minimise/minimize ... reboiler duty), keeping
            # the model's original wording appended; anything else still fails closed.
            if isinstance(s, dict) and isinstance(s.get("objective"), str):
                _o = s["objective"]
                if ("reboiler_duty" not in _o
                        and re.search(r"minimi[sz]e\b[\s\S]{0,60}?reboiler[\s_]+duty", _o, re.I)):
                    s["objective"] = "minimize reboiler_duty -- " + _o
            if isinstance(s, dict):
                dvs = s.get("decision_variables")
                if isinstance(dvs, dict):
                    dvs = [{**(v if isinstance(v, dict) else {}), "name": k}
                           for k, v in dvs.items()]
                if isinstance(dvs, list):
                    out = []
                    for dv in dvs:
                        if isinstance(dv, str):
                            if dv in canon:
                                out.append(dict(canon[dv]))
                            continue
                        if isinstance(dv, dict):
                            d = dict(dv)
                            if "lower_bound" not in d and "lower" in d:
                                d["lower_bound"] = d["lower"]
                            if "upper_bound" not in d and "upper" in d:
                                d["upper_bound"] = d["upper"]
                            base = canon.get(d.get("name"))
                            if base:
                                for k in ("type", "lower_bound", "upper_bound"):
                                    d.setdefault(k, base[k])
                            out.append(d)
                    s["decision_variables"] = out
    # raw_text is the model's VERBATIM accumulated text -- the capture source of truth.
    return spec, tool_log, text


if __name__ == "__main__":
    # Standalone stdio MCP server (interoperability): real Aspen Plus V14 backend only
    # (the analytical surrogate has been removed; requires Aspen Plus available).
    from case2_flowsheet.flowsheet_copilot import AspenFlowsheet, RAGEngine, CONFIG
    _fs = AspenFlowsheet(CONFIG, mock=False)
    _rag = RAGEngine()
    _srv = make_grounding_server(_fs, _rag, feed_z=None)
    _srv.run()
