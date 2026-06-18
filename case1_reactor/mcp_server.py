# -*- coding: utf-8 -*-
"""
Real MCP server for Case Study 1 (exothermic CSTR) grounding tools, plus the
tool-use proposer that drives them.

Exposes ONLY *advisory grounding* tools to the LLM (symmetric with case2_flowsheet):
  - retrieve_constraints(query, top_k): CURRENT versioned, provenance-tracked operating
                                 procedures / equipment limits (SOPs; constraint channel)
  - retrieve_knowledge(query, top_k):   design-rationale / engineering-knowledge passages
                                 (the rationale tier; knowledge channel)
  - simulate_setpoint(Tc):       READ-ONLY worst-case peak-temperature check on the
                                 trusted CSTR model, computed with the SAME physics
                                 as the gate's safety screen (cstr.simulate over the
                                 worst-case degraded plant; T_run / safety margin).

The optimizer, action-selection, and the admissibility certificate are NOT exposed
-- they remain harness-owned, so the LLM can ground its proposal but the verifier
keeps sole authority over the committed action (the framework's safety invariant).

Standalone stdio MCP server (cross-process interoperability):
    python -m case1_reactor.mcp_server
"""
import json
import os

from case1_reactor.core import mcp_bridge

T_RUN_DEFAULT       = 475.0   # runaway limit (K); matches TABLE2 / the gate
SAFE_MARGIN_DEFAULT = 5.0     # K; matches SAFE_MARGIN
TOOLUSE_MAX_TOKENS  = 8000    # per-call output cap for the tool-use proposer. 1500 -> 3000 after the
                              # 2026-06-09 smoke (final turns truncated: ~700 tokens of analysis
                              # before the JSON; max observed final 1858) -> 4096 (2026-06-10) ->
                              # 8000 (matches CS2 for symmetry; the two-retrieval-tool proposer now
                              # carries more grounding context). Billing is by ACTUAL tokens, so the
                              # extra headroom costs nothing when unused; caps must never bind.
HORIZON_DEFAULT     = 5.0     # min; MUST match the controller's screen horizon so this tool's
                              # peak-T matches the gate's screen. Advisory only: the gate
                              # re-verifies independently, so a mismatch degrades grounding
                              # quality, not safety. Update if the controller horizon changes.


def make_grounding_server(cstr, rag, Ca, T, UAf_screen, Ca0f_screen,
                          horizon=HORIZON_DEFAULT, T_run=T_RUN_DEFAULT,
                          safe_margin=SAFE_MARGIN_DEFAULT, name="axiom-grounding",
                          expose_retrieval=True):
    """FastMCP server whose tools close over the live decision context.

    expose_retrieval=True (default): [retrieve_constraints, retrieve_knowledge, simulate_setpoint]
    -- the same two-retrieval-tool + simulate layout as case2_flowsheet. expose_retrieval=False
    (+MCP-only ablation cell): NO retrieval tools are registered, so the proposer can ground its
    ACTION in computed physics (simulate_setpoint) but has no knowledge-retrieval tool --
    isolating MCP's standalone contribution from RAG's."""
    import numpy as np

    def retrieve_constraints(query: str, top_k: int = 4) -> str:
        """Retrieve the CURRENT provenance-tracked operating procedures and equipment limits (SOPs) relevant to the query, with source ids and versions. Cite ONLY ids returned here; never invent a limit."""
        hits = rag.retrieve(query, top_k=top_k, tiers=("operating-constraint",))
        return json.dumps([{"id": h["id"], "title": h.get("title"),
                            "text": h.get("text"), "source": h.get("source"),
                            "version": h.get("version")} for h in hits])

    def retrieve_knowledge(query: str, top_k: int = 4) -> str:
        """Retrieve provenance-tracked design-rationale / engineering-knowledge passages (e.g. runaway criteria, kinetics rationale, heuristics) relevant to the query."""
        hits = rag.retrieve(query, top_k=top_k, tiers=("rationale",))
        return json.dumps([{"id": h["id"], "title": h.get("title"),
                            "text": h.get("text"), "source": h.get("source"),
                            "version": h.get("version")} for h in hits])

    def simulate_setpoint(Tc: float) -> str:
        """Predict the worst-case peak reactor temperature for a candidate coolant setpoint Tc (Kelvin) on the trusted CSTR model under the current degraded (worst-case) plant. Returns peak T, the runaway limit, the margin, and whether it clears the limit with the required safety margin. READ-ONLY: commits nothing."""
        try:
            sol = cstr.simulate((0, horizon), [Ca, T], Tc=float(Tc),
                                UAf=UAf_screen, Ca0f=Ca0f_screen, n_eval=200)
            peak = float(np.max(sol.y[1]))
        except Exception as exc:
            return json.dumps({"error": "simulation_failed: %s" % exc})
        return json.dumps({"Tc_K": round(float(Tc), 2),
                           "peak_T_K": round(peak, 2),
                           "T_runaway_K": T_run,
                           "margin_K": round(T_run - peak, 2),
                           "clears_runaway_with_margin": bool(T_run - peak >= safe_margin)})

    tools = ([retrieve_constraints, retrieve_knowledge, simulate_setpoint] if expose_retrieval
             else [simulate_setpoint])
    return mcp_bridge.make_server(name, tools)


TOOLUSE_PROPOSER_SYSTEM = (
    "You are the planning layer of a safety-critical CSTR decision-support system. "
    "You have tools: simulate_setpoint(Tc) checks a candidate coolant setpoint's worst-case "
    "peak temperature against the runaway limit on the trusted model, retrieve_constraints(query) "
    "fetches the CURRENT provenance-tracked operating procedures and equipment limits (SOPs), and "
    "retrieve_knowledge(query) fetches design-rationale passages. GROUND your proposal: check "
    "candidate setpoints with simulate_setpoint, take the limits you rely on from "
    "retrieve_constraints (do NOT invent them), and cite ONLY ids returned by retrieve_constraints "
    "or retrieve_knowledge. The HARD runaway limit and all safety screening are enforced DOWNSTREAM "
    "by the process model and are NOT yours to set; your proposal can only inform, never override, "
    "the verified controller. When finished, return ONLY a JSON object with keys "
    "'hypothesis_weighting', 'candidate_actions' (a JSON list of plain NUMBERS -- advisory "
    "coolant setpoints in Kelvin; no objects, no labels -- put any rationale in 'procedure'), "
    "and 'contextual_constraints' {operating_envelope (the key temperature limits in K you rely "
    "on, taken from retrieve_constraints -- do NOT invent them), procedure, citations}."
)


TOOLUSE_PROPOSER_SYSTEM_MCP_ONLY = (
    "You are the planning layer of a safety-critical CSTR decision-support system. "
    "You have ONE tool: simulate_setpoint(Tc) checks a candidate coolant setpoint's "
    "worst-case peak temperature against the runaway limit on the trusted model. GROUND "
    "your proposal by checking candidate setpoints with simulate_setpoint. You have NO "
    "document-retrieval tools, so do NOT cite sources -- return an empty 'citations' list. "
    "The HARD runaway limit and all safety screening are enforced DOWNSTREAM by the process "
    "model and are NOT yours to set; your proposal can only inform, never override, the "
    "verified controller. When finished, return ONLY a JSON object with keys "
    "'hypothesis_weighting', 'candidate_actions' (a JSON list of plain NUMBERS -- advisory "
    "coolant setpoints in Kelvin; no objects, no labels -- put any rationale in 'procedure'), "
    "and 'contextual_constraints' {operating_envelope (the key temperature limits in K you "
    "believe apply; you have NO retrieval source for them), procedure, citations}."
)


def tooluse_decision_spec(client, model, query, top_hyps, enkf_summary,
                          cstr, rag, Ca, T, UAf_screen, Ca0f_screen,
                          horizon=HORIZON_DEFAULT, mcp_only=False):
    """Tool-use proposer: the LLM grounds its proposal by calling the MCP grounding
    tools, then returns a decision spec parsed by the SAME parse_decision_spec_json as
    the one-shot path. Returns (spec_dict, tool_call_log). `client` is an Anthropic-style
    client (real or the scripted mock from mcp_bridge).

    mcp_only=True (+MCP-only ablation cell): expose simulate_setpoint ONLY (no retrieval
    tool) and use the no-citation system prompt, isolating MCP's standalone contribution.

    Returns (spec_dict, tool_call_log, raw_text): raw_text is the model's VERBATIM
    accumulated text (the capture source of truth -- logged so a parse failure stays
    diagnosable); the spec is parsed from the LAST complete JSON object in it
    (mcp_bridge.extract_last_json) via the SAME parse_decision_spec_json as one-shot."""
    from case1_reactor.axiom import parse_decision_spec_json
    server = make_grounding_server(cstr, rag, Ca, T, UAf_screen, Ca0f_screen,
                                   horizon=horizon, expose_retrieval=not mcp_only)
    system = TOOLUSE_PROPOSER_SYSTEM_MCP_ONLY if mcp_only else TOOLUSE_PROPOSER_SYSTEM
    hyp_keys = [h[0] for h in top_hyps]
    user = ("QUERY: %s\nENKF STATE: %s\nFAULT HYPOTHESES (use these keys only): %s\n"
            "Ground your proposal with the tools, then return the JSON object."
            % (query, enkf_summary, ", ".join(hyp_keys)))
    text, tool_log = mcp_bridge.run_grounded(
        client, model, system, user, server,
        max_tokens=TOOLUSE_MAX_TOKENS)
    spec = parse_decision_spec_json(mcp_bridge.extract_last_json(text) or text)
    spec = canonicalize_candidate_actions(spec)
    return spec, tool_log, text


# Setpoint keys the live model used when packing candidates as objects (2026-06-10 smoke:
# dict-of-label, list-of-objects, and {"setpoints_K": [...], "recommended_K": ...} shapes --
# every candidate carried simulate-grounded Tc_K values; only the PACKAGING varied).
_SETPOINT_KEYS = ("Tc_K", "Tc", "setpoint_K", "setpoint", "recommended_K", "value")


def canonicalize_candidate_actions(spec):
    """Shape-only canonicalization of `candidate_actions` to the locked one-shot shape (a flat
    list of Kelvin numbers). Extracts the model's OWN stated setpoints from the object/dict
    packagings observed live; nothing is invented -- entries with no recognizable setpoint are
    dropped and anything else fails closed in the unchanged validate_decision_spec."""
    if not isinstance(spec, dict):
        return spec
    ca = spec.get("candidate_actions")

    def _nums_from(obj):
        out = []
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            out.append(float(obj))
        elif isinstance(obj, str):
            try:
                out.append(float(obj))
            except ValueError:
                pass
        elif isinstance(obj, dict):
            if isinstance(obj.get("setpoints_K"), list):
                for v in obj["setpoints_K"]:
                    out.extend(_nums_from(v))
            for k in _SETPOINT_KEYS:
                v = obj.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out.append(float(v))
        return out

    nums = []
    if isinstance(ca, list):
        for entry in ca:
            nums.extend(_nums_from(entry))
    elif isinstance(ca, dict):
        if isinstance(ca.get("setpoints_K"), list) or any(k in ca for k in _SETPOINT_KEYS):
            nums.extend(_nums_from(ca))
        else:                                   # dict-of-label -> {Tc_K: ...} objects
            for v in ca.values():
                nums.extend(_nums_from(v))
    if nums:
        seen, dedup = set(), []
        for n in nums:
            if n not in seen:
                seen.add(n)
                dedup.append(n)
        spec["candidate_actions"] = dedup
    return spec


if __name__ == "__main__":
    # Standalone stdio MCP server (interoperability). Fresh real objects at a nominal,
    # fixed sample state; tools are stateless w.r.t. the harness in this mode.
    from case1_reactor.axiom import CSTRModel, RAGEngine, SOP_LIBRARY
    _cstr = CSTRModel()
    _rag = RAGEngine(SOP_LIBRARY, use_dense=bool(os.environ.get("RAG_DENSE")))
    _srv = make_grounding_server(_cstr, _rag, Ca=0.5, T=440.0,
                                 UAf_screen=0.5, Ca0f_screen=1.2)
    _srv.run()
