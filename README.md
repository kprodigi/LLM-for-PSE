# LLM-for-PSE: Admissibility-Aware, Fail-Closed Decision Support

Reference implementation for **"Admissibility-Aware AI for Fail-Closed Decision Support in Process Systems Engineering"** (Sarker and Kazi, *Systems and Control Transactions* (PSE Press), FOCAPO-CPC 2027).

A large language model is restricted to **proposing and explaining** decisions; a deterministic **admissibility check** on a trusted process model owns every release. The model is grounded from two sides: retrieval-augmented generation (RAG) supplies source-traceable knowledge, and the Model Context Protocol (MCP) supplies trusted first-principles simulators. A decision is released only when it passes well-formedness, feasibility, robustness, evidence, recommendation-safety, consistency, and a domain check; otherwise the system fails closed to a verified safe action or abstains.

## Contents

```
case1_reactor/            Case study 1 - exothermic CSTR control
  axiom.py                  end-to-end pipeline (reactor model, RAG, certificate, fail-closed gate)
  mcp_server.py             MCP tool server
  core/                     shared kernel (env, tools, diagnostics, MCP bridge, embeddings)
case2_flowsheet/          Case study 2 - methanol-water distillation design (Aspen Plus V14)
  flowsheet_copilot.py      end-to-end pipeline (column model, optimizer, certificate, counterfactual)
  mcp_server.py             MCP tool server
  core/                     shared kernel
  aspen/                    Aspen Plus V14 baseline builder (build_baseline.py, baseline_meoh_water.inp)
```

## Setup

```bash
pip install -r requirements.txt
```

One file covers both case studies. `pywin32` (case study 2's Aspen Plus V14 COM bridge) installs automatically on Windows only, so the same command works on any platform.

## Run

```bash
# Case study 1 - any platform, mock LLM, no API key
AXIOM_MOCK_LLM=1 python case1_reactor/axiom.py

# Case study 2 - Windows + Aspen Plus V14; build the baseline once, then run
python case2_flowsheet/aspen/build_baseline.py
FCO_MOCK_LLM=1 python case2_flowsheet/flowsheet_copilot.py
```

The live-LLM path reads `ANTHROPIC_API_KEY` from the environment (proposer and explainer use Claude Sonnet 4.6; the weak-model arm uses Claude Haiku 4.5). Case study 2 uses the real Aspen Plus V14 COM backend.

## Citation

> Sarker, N., and Kazi, M.-K. (2027). Admissibility-Aware AI for Fail-Closed Decision Support in Process Systems Engineering. *Systems and Control Transactions* (PSE Press), FOCAPO-CPC 2027.

## License

MIT - see [LICENSE](LICENSE).
