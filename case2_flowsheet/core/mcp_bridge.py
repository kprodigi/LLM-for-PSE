# -*- coding: utf-8 -*-
"""
Real MCP tool-use bridge for the grounded reasoning (proposer) layer.

This turns the soft layer from a ONE-SHOT text call into a TOOL-USING agent that
grounds its proposal by calling trusted PSE tools over the *real* Model Context
Protocol. We use the MCP SDK's in-memory transport (genuine MCP/JSON-RPC messages
between a FastMCP server and a client session in the same process) so the tools
can close over the live decision context (current state, ensemble, corpus) while
remaining real MCP. Each case also ships a stdio entrypoint (its mcp_server.py)
exposing the same tools cross-process, for interoperability.

SAFETY INVARIANT: only *advisory grounding* tools are exposed here -- retrieval
and READ-ONLY simulation. The optimizer, action-selection, and the admissibility
certificate are NEVER exposed, so the LLM grounds its proposal with tools while
the verifier keeps sole authority over the committed decision.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace


def make_server(name, fns):
    """Build a FastMCP server exposing `fns` (typed callables) as MCP tools.
    FastMCP infers each tool's name/description/input-schema from the function's
    name, docstring, and type-hinted signature."""
    from mcp.server.fastmcp import FastMCP
    app = FastMCP(name)
    for fn in fns:
        app.add_tool(fn)
    return app


def _tools_to_anthropic(listed):
    """Convert an MCP list_tools() result into Anthropic tool-use definitions."""
    return [{"name": t.name,
             "description": t.description or "",
             "input_schema": t.inputSchema}
            for t in listed.tools]


def _result_text(call_result):
    """Flatten an MCP call_tool() result into a plain string for the model."""
    parts = []
    for block in getattr(call_result, "content", []) or []:
        txt = getattr(block, "text", None)
        parts.append(txt if txt is not None else str(block))
    return "\n".join(parts) if parts else ""


# Harness recovery notice for tool-budget exhaustion (NOT part of the frozen proposer prompts;
# injected only when the model is still requesting tools on its final permitted call). Without
# it a heavily-probing model can spend every iteration on tools and never emit the final JSON
# (observed live: CS2 smoke, 17 tool calls in 6 turns -> empty spec -> safe abstention).
FINAL_ANSWER_NUDGE = ("Tool budget exhausted. Using the evidence gathered so far, return ONLY "
                      "the final JSON object now -- no further tool calls, no prose.")


async def _loop(client, model, system, user, server, max_iters, max_tokens, tool_log):
    from mcp.shared.memory import create_connected_server_and_client_session
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        listed = await session.list_tools()
        atools = _tools_to_anthropic(listed)
        messages = [{"role": "user", "content": user}]
        final_text = ""
        for i in range(max_iters):
            kw = dict(model=model, max_tokens=max_tokens, temperature=0.0,
                      system=system, tools=atools, messages=messages)
            if i == max_iters - 1:
                # last permitted call: force a text answer (no further tool calls), so the
                # decision spec is always emitted within the pre-registered call budget.
                kw["tool_choice"] = {"type": "none"}
            resp = client.messages.create(**kw)
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    final_text += block.text
                elif btype == "tool_use":
                    out = _result_text(await session.call_tool(block.name, block.input or {}))
                    tool_log.append({"tool": block.name, "input": block.input, "output": out})
                    tool_results.append({"type": "tool_result",
                                         "tool_use_id": block.id, "content": out})
            if getattr(resp, "stop_reason", None) == "tool_use" and tool_results:
                nxt = list(tool_results)
                if i == max_iters - 2:        # entering the forced-final round: tell the model
                    nxt.append({"type": "text", "text": FINAL_ANSWER_NUDGE})
                messages.append({"role": "user", "content": nxt})
                continue
            break
        return final_text, tool_log


def run_grounded(client, model, system, user, server, max_iters=6, max_tokens=1200):
    """Run the tool-use loop to completion; return (final_text, tool_call_log).
    `client` is an Anthropic-style client (real or scripted) exposing
    messages.create(...). Synchronous wrapper around the async MCP session."""
    tool_log = []
    return asyncio.run(_loop(client, model, system, user, server,
                             max_iters, max_tokens, tool_log))


def extract_last_json(text):
    """Return the LAST balanced {...} substring of `text` that json.loads accepts, or None.

    The tool-use loop accumulates text across turns, so prose (and brace noise) can precede
    the final JSON spec, and a max_tokens-truncated final turn leaves an unbalanced tail --
    the greedy first-{-to-last-} regex then fails even when a complete object exists earlier
    in the text. Scanning candidates from the END recovers the spec whenever any complete
    JSON object was produced; with no complete object this returns None (or an inner object
    the downstream schema validation rejects), preserving fail-closed behavior."""
    s = text or ""
    end = len(s)
    while True:
        close = s.rfind("}", 0, end)
        if close < 0:
            return None
        depth, i = 0, close
        while i >= 0:
            c = s[i]
            if c == "}":
                depth += 1
            elif c == "{":
                depth -= 1
                if depth == 0:
                    cand = s[i:close + 1]
                    try:
                        json.loads(cand)
                        return cand
                    except Exception:
                        break              # not valid JSON -> try an earlier closing brace
            i -= 1
        end = close


# --------------------------------------------------------------------------- #
# Offline validation: a scripted "LLM" that drives the tool-use loop with NO   #
# API call, so the MCP round-trip + loop are testable for $0.                  #
# --------------------------------------------------------------------------- #
class ScriptedLLM:
    """Mimics anthropic.Anthropic().messages.create for offline tests. `script`
    is a list of turns; each turn is a list of blocks, where a block is either
    {'text': ...} or {'tool': name, 'input': {...}}. The last turn should be
    text-only (the final answer)."""
    def __init__(self, script):
        self._script = list(script)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        turn = self._script.pop(0) if self._script else [{"text": ""}]
        content, has_tool = [], False
        for b in turn:
            if "tool" in b:
                has_tool = True
                content.append(SimpleNamespace(type="tool_use", id="t_%d" % len(content),
                                               name=b["tool"], input=b.get("input", {})))
            else:
                content.append(SimpleNamespace(type="text", text=b.get("text", "")))
        return SimpleNamespace(content=content,
                               stop_reason="tool_use" if has_tool else "end_turn")


if __name__ == "__main__":   # offline self-test: real MCP round-trip, no API
    def retrieve_sops(query: str, top_k: int = 4) -> str:
        """Retrieve provenance-tracked SOP chunks relevant to a query."""
        return "[SOP-026] Operating envelope: coolant Tc adjustable 250-345 K (nominal 300)."

    def simulate_setpoint(Tc: float) -> str:
        """Simulate a candidate coolant setpoint; return predicted peak T and whether it is safe."""
        peak = 470.0 - (300.0 - float(Tc)) * 0.5
        return "peak_T=%.1f K; clears_runaway=%s" % (peak, peak < 470.0)

    srv = make_server("axiom-grounding-selftest", [retrieve_sops, simulate_setpoint])
    script = [
        [{"tool": "simulate_setpoint", "input": {"Tc": 290.0}}],
        [{"tool": "retrieve_sops", "input": {"query": "coolant actuator range", "top_k": 2}}],
        [{"text": "Proposed Tc=290 K, grounded by a tool check and SOP-026."}],
    ]
    txt, log = run_grounded(ScriptedLLM(script), "mock-model", "sys",
                            "Propose a safe coolant setpoint.", srv)
    print("TOOLS CALLED:", [e["tool"] for e in log])
    print("TOOL OUTPUTS:", [e["output"] for e in log])
    print("FINAL TEXT:", txt)
    assert [e["tool"] for e in log] == ["simulate_setpoint", "retrieve_sops"]
    print("MCP bridge self-test: OK")
