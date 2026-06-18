# -*- coding: utf-8 -*-
"""
Shared diagnostics-logger factory.

Both case studies record solver/tool failures to a per-run diagnostics.log with
the same handler + formatter + no-propagate setup; only the logger NAME, file
PATH, and LEVEL differ (parameters here). Behaviour is identical to the previous
inline setup in each script.
"""
import logging


def make_diagnostics_logger(name, log_path, level=logging.WARNING):
    """Return a file-backed diagnostics logger (handlers added once, idempotent).

    name      logger name (e.g. "axiom.diagnostics", "fco.diagnostics")
    log_path  destination file (e.g. <results>/diagnostics.log)
    level     logging level (case 1 used WARNING, case 2 used INFO)
    """
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(level)
        log.propagate = False
    return log
