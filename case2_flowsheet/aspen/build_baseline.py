# -*- coding: utf-8 -*-
"""
Reproducible builder for the Case-Study-2 real-Aspen baseline (R0).

Loads the CANONICAL, deterministic Aspen input language
(case2_flowsheet/aspen/baseline_meoh_water.inp — tracked in git, sha-256 anchored)
over COM, runs it, verifies convergence + the starting specs, and saves the baseline
.bkp at CONFIG["system"]["baseline_bkp"].

The .inp is the single source of truth for the baseline: it is byte-stable across
rebuilds, so the baseline reconstructs from git WITHOUT depending on the .bkp. A .bkp
is a binary archive embedding a save-timestamp/host metadata, so ITS hash is NOT
reproducible (verified: two identical rebuilds produced different .bkp hashes). Hence
reproducibility is verified against the .inp sha-256 + the converged outputs
(duty/xD/xB), never the .bkp hash. The heavy .bkp is kept out of git and backed up.

If the canonical .inp is ever missing it is bootstrapped from the embedded INP_TEXT
and then checked against the sha anchor, so the two can never silently drift.

Verified facts baked in (see BUILD_NOTES.md R0):
  * Early binding via gencache.EnsureDispatch is REQUIRED (late binding cannot
    invoke Engine.Run2 on this multi-interface object).
  * Run on the MAIN thread (a worker thread -> 'CoInitialize has not been called').
  * SI units -> reboiler duty in Watt, temperatures in degC, flows in kmol/h.

Usage:  python build_baseline.py        (real Aspen Plus V14 required; ~10-20 s)
Exits non-zero if the .inp anchor mismatches, or the baseline does not converge /
misses the starting specs.
"""
import os
import sys
import time
import shutil
import hashlib

os.environ.setdefault("PYTHONUTF8", "1")
# repo root on sys.path so `case2_flowsheet` imports when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ASPEN_DIR = os.path.dirname(os.path.abspath(__file__))
# Canonical, deterministic baseline input language (the single source of truth).
CANON_INP = os.path.join(ASPEN_DIR, "baseline_meoh_water.inp")
# Deterministic anchor: sha-256 of the 563-byte canonical .inp. THIS is the
# reproducibility check that holds across machines/rebuilds (unlike the .bkp hash).
BASELINE_INP_SHA256 = "027c3cef5c8d555019dce651b2858d88396c75621dcddeb9a75d591cd14f1b0d"

# Embedded copy of the canonical input language, used ONLY to bootstrap CANON_INP if
# the tracked file is missing; the sha anchor above guards against any drift.
INP_TEXT = """;Methanol-Water continuous RadFrac baseline (Case Study 2)

IN-UNITS SI MOLE-FLOW='kmol/hr' PRESSURE=bar TEMPERATURE=C

DEF-STREAMS CONVEN ALL

COMPONENTS
    METHANOL CH4O /
    WATER H2O

FLOWSHEET
    BLOCK COL IN=FEED OUT=DIST BOT

PROPERTIES NRTL

STREAM FEED
    SUBSTREAM MIXED VFRAC=0. PRES=1.013
    MOLE-FLOW METHANOL 50. / WATER 50.

BLOCK COL RADFRAC
    PARAM NSTAGE=25
    COL-CONFIG CONDENSER=TOTAL REBOILER=KETTLE
    FEEDS FEED 13
    PRODUCTS DIST 1 L / BOT 25 L
    P-SPEC 1 1.013
    COL-SPECS MOLE-D=50. MOLE-RR=2.5
"""


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    from case2_flowsheet import flowsheet_copilot as M  # configured paths / thresholds
    from win32com.client import gencache

    # 1) Canonical .inp = source of truth. Bootstrap from INP_TEXT if absent, then
    #    verify the deterministic sha anchor so the .inp can never silently drift.
    if not os.path.exists(CANON_INP):
        with open(CANON_INP, "w") as fh:
            fh.write(INP_TEXT)
        print("bootstrapped canonical .inp:", CANON_INP)
    got = _sha256(CANON_INP)
    if got != BASELINE_INP_SHA256:
        print("FAILED: canonical .inp sha-256 mismatch (refusing to build)\n"
              "  expected %s\n  got      %s" % (BASELINE_INP_SHA256, got))
        return 2
    print("canonical .inp:", CANON_INP)
    print("  sha256 %s (deterministic anchor OK)" % got)

    bkp = M.CONFIG["system"]["baseline_bkp"]
    work_inp = os.path.splitext(bkp)[0] + ".inp"
    os.makedirs(os.path.dirname(bkp), exist_ok=True)
    # Work from a byte-exact copy in results/ (Aspen scatters scratch beside the
    # loaded .inp; keep that out of the tracked aspen/ directory).
    shutil.copyfile(CANON_INP, work_inp)

    t0 = time.time()
    asp = gencache.EnsureDispatch("Apwn.Document")
    try:
        asp.SuppressDialogs = 1
    except Exception as err:
        print("  (SuppressDialogs failed: %s)" % err)
    asp.InitFromArchive2(os.path.abspath(work_inp))
    try:
        asp.Visible = False
    except Exception:
        pass
    asp.Engine.Run2()
    dt = time.time() - t0

    def val(path):
        n = asp.Tree.FindNode(path)
        return None if n is None or n.Value is None else n.Value

    blk = val(r"\Data\Blocks\COL\Output\BLKSTAT")
    duty_kw = (val(r"\Data\Blocks\COL\Output\REB_DUTY") or 0) / 1000.0
    xd = val(r"\Data\Streams\DIST\Output\MOLEFRAC\MIXED\METHANOL")
    xb = val(r"\Data\Streams\BOT\Output\MOLEFRAC\MIXED\METHANOL")
    print("ran in %.1fs | BLKSTAT=%s duty=%.1f kW xD=%.5f xB=%.2e" %
          (dt, blk, duty_kw, xd or -1, xb or -1))

    converged = str(blk) in ("0", "0.0")
    specs_ok = (xd is not None and xd >= 0.99) and (xb is not None and xb <= 0.01)
    if not (converged and specs_ok):
        print("FAILED: converged=%s specs_ok=%s -> not saving" % (converged, specs_ok))
        try: asp.Close()
        except Exception: pass
        return 1

    asp.SaveAs(os.path.abspath(bkp))
    print("SAVED baseline:", bkp)
    # The .bkp hash is PROVENANCE ONLY (non-reproducible); the anchor is the .inp
    # sha above + these converged outputs. Printed so the provenance is on record.
    try:
        print("  .bkp sha256 (provenance, non-reproducible):", M._file_sha256(bkp))
    except Exception as err:
        print("  (hash failed: %s)" % err)
    try: asp.Close()
    except Exception: pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
