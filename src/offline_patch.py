"""Network mode switch for HF / timm downloads.

Default is ONLINE via the hf-mirror (verified 2026-10-24: 358MB/31s).
Opt out with any of:

  python src/unified_train_diag.py --offline ...
  DIAG_ONLINE=0 python src/unified_train_diag.py ...
  export HF_HUB_OFFLINE=1 TIMM_OFFLINE=1  (explicit user export wins)

Online mode defaults HF_ENDPOINT to the hf-mirror when the user did
not set one, per admin notice (no SSH-tunnel downloads over the ALiYun
line; use mirror, clash-for-linux only if mirror fails).

Call apply() once at startup; importing this module auto-applies with
online=None (i.e. auto-detect from env/argv, default online).
"""
import os
import sys

MIRROR = "https://hf-mirror.com"

OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "TIMM_OFFLINE", "HF_DATASETS_OFFLINE")


def _argv_wants_online():
    return "--online" in sys.argv


def _argv_wants_offline():
    return "--offline" in sys.argv


def is_online_requested(online=None):
    """Resolve online flag: explicit arg > explicit env > argv > default on."""
    if online is not None:
        return bool(online)
    if _argv_wants_offline():
        return False
    if _argv_wants_online():
        return True
    diag = os.environ.get("DIAG_ONLINE", "").strip()
    if diag == "0":
        return False
    if diag == "1":
        return True
    # explicit user exports win over the default.
    vals = {os.environ.get(k, "").strip() for k in OFFLINE_FLAGS}
    if "1" in vals:
        return False
    if "0" in vals:
        return True
    return True


def apply(online=None):
    # explicit True/False forces the mode; None = auto-detect (default on).
    on = is_online_requested(online)
    if on:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TIMM_OFFLINE"] = "0"
        os.environ["HF_DATASETS_OFFLINE"] = "0"
        os.environ.setdefault("HF_ENDPOINT", MIRROR)
        os.environ.setdefault("HF_HUB_VERBOSITY", "info")
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TIMM_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    return on


def describe():
    return {
        "online": os.environ.get("HF_HUB_OFFLINE") != "1",
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
        "TIMM_OFFLINE": os.environ.get("TIMM_OFFLINE", ""),
    }


# import-time side effect (keeps old `import offline_patch` call working)
apply()
