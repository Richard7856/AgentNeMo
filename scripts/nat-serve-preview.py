"""NAT REST API launcher for preview_start / Claude Code environments.

WHY this file exists instead of running `nat serve` directly:
  The project lives in an iCloud-synced directory. When iCloud's filecoordinationd
  daemon is actively syncing, it places coordination locks on egg-info files.
  Python's importlib.metadata.entry_points() reads ALL installed distributions
  (including insurance_claims_agent.egg-info/entry_points.txt) and blocks
  indefinitely waiting for the lock to clear.
  `make serve` from a terminal works because the shell environment already has
  iCloud coordination resolved. preview_start hits the lock on cold launch.

FIX — two parts that MUST work together:
  1. Patch PathDistribution.read_text to return hardcoded content for our
     egg-info without touching the file on disk (bypasses the iCloud lock).
  2. Invoke NAT's CLI directly in this same process instead of os.execv.
     os.execv replaces the process image, losing the monkey-patch above.
"""
import os
import sys

PROJECT_DIR = "/Users/richardfigueroa/Library/Mobile Documents/com~apple~CloudDocs/Nvidia/insurance-claims-agent"

os.chdir(PROJECT_DIR)

# Prepend project root so the editable install resolves insurance_claims.*
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, ".venv/lib/python3.12/site-packages"))

# Load env vars from /tmp copy — preview_start sandbox cannot source .env
# directly from the iCloud path.
with open("/tmp/nat_agent.env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

os.environ["PYTHONPATH"] = PROJECT_DIR

# ── iCloud lock bypass ──────────────────────────────────────────────────────
# Must happen BEFORE any `import nat` because entrypoint.py calls
# discover_and_load_cli_plugins() at module level, which triggers entry_points().
#
# PathDistribution.read_text(filename) is the single choke-point: every call
# to importlib.metadata.entry_points() (with or without a group argument)
# eventually calls read_text('entry_points.txt') on each distribution.
# We intercept only our own egg-info and return the known content as a string.
import importlib.metadata as _im

# Exact content of insurance_claims_agent.egg-info/entry_points.txt
_INSURANCE_EP_TXT = "[nat.components]\ninsurance_claims = insurance_claims.register\n"

_orig_read_text = _im.PathDistribution.read_text


def _patched_read_text(self, filename):
    # Short-circuit only for our egg-info — all other distributions go through
    # the original method unchanged.
    if filename == "entry_points.txt":
        try:
            if "insurance_claims_agent" in str(self._path):
                return _INSURANCE_EP_TXT
        except AttributeError:
            pass
    return _orig_read_text(self, filename)


_im.PathDistribution.read_text = _patched_read_text
# ────────────────────────────────────────────────────────────────────────────

# Import NAT CLI AFTER the patch is in place.
from nat.cli.entrypoint import cli  # noqa: E402

sys.argv = [
    "nat",
    "serve",
    "--config_file",
    os.path.join(PROJECT_DIR, "configs/config.yml"),
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
]

cli()
