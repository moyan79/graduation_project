"""Call once at startup: fail fast on cached weights, avoid long HF retries when offline."""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TIMM_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
