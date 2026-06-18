# -*- coding: utf-8 -*-
"""
Shared MCP-STYLE in-process tool registry.

Identical dispatch + call-logging behaviour across both case studies. The only
drift was the logged-args truncation length (case 1 used 100, case 2 used 120),
kept here as the `args_maxlen` parameter so each case's audit-trace 'args'
strings remain byte-identical to before.
"""
import time
from collections import defaultdict

import numpy as np


class MCPToolRegistry:
    """
    In-process tool-dispatch table (name->callable) with call logging/timing, used by
    the HARNESS for its own verification/optimization tools. This is NOT the Model
    Context Protocol and is intentionally NOT exposed to the LLM.

    The LLM-facing GROUNDING tools (retrieve_* + simulate_*) ARE served over real MCP
    — see each case's mcp_server.py (FastMCP server + in-memory transport) and
    core/mcp_bridge.py. Keeping the harness's verification/optimization tools off that
    MCP surface is deliberate: the LLM grounds its proposal via MCP, but the gate
    keeps sole authority (the framework's safety invariant).
    """
    def __init__(self, args_maxlen=120):
        self._tools   = {}
        self.call_log = []
        self.args_maxlen = args_maxlen

    def register(self, name, fn, description, schema=None):
        self._tools[name] = {"fn": fn, "description": description,
                             "schema": schema or {}, "calls": 0}

    def call(self, name, **kwargs):
        assert name in self._tools, f"Tool '{name}' not registered"
        t0  = time.time()
        result = self._tools[name]["fn"](**kwargs)
        self._tools[name]["calls"] += 1
        self.call_log.append({"tool": name,
                              "latency_ms": round((time.time() - t0) * 1000, 1),
                              "args": str(kwargs)[:self.args_maxlen]})
        return result

    def timing(self):
        d = defaultdict(list)
        for e in self.call_log:
            d[e["tool"]].append(e["latency_ms"])
        return {k: {"mean": round(float(np.mean(v)), 1), "total": len(v)}
                for k, v in d.items()}
