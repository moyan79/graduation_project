"""Build any of the 15 sub-domain models through the sibling ForensicHub registry.

Sibling layout (local == remote):
  graduation_project/
    ForensicHub/                 <- ForensicHub repo
    linear_probe-and-gradient/   <- this project (you are here)

So ForensicHub is found as <project_root>/../ForensicHub.
"""
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[2]          # graduation_project/
FORENSICHUB_REPO = PROJECT_ROOT / "ForensicHub"
for p in (str(FORENSICHUB_REPO),):
    if p not in sys.path:
        sys.path.insert(0, p)

# short key (cli) -> ForensicHub build config
# image_size / norm follow the corresponding statics/*.yaml conventions.
MODEL_CFG = {
    "resnet":       dict(registry="Resnet101",      image_size=256, norm="imagenet",
                         init=dict(output_type="label", image_size=256)),
    "efficientnet": dict(registry="EfficientB4",    image_size=256, norm="imagenet",
                         init=dict(output_type="label", image_size=256)),
    "capsulenet":   dict(registry="Capsule_net",    image_size=256, norm="imagenet",
                         init=dict(), need_deepfake_yaml="capsule_net.yaml"),
    "segformer":    dict(registry="Segformerb3",     image_size=256, norm="imagenet",
                         init=dict(output_type="label", image_size=256)),
    "swin":         dict(registry="SwinSmall",       image_size=256, norm="imagenet",
                         init=dict(output_type="label", image_size=256)),
    "convnext":     dict(registry="ConvNextSmall",   image_size=256, norm="imagenet",
                         init=dict(output_type="label", image_size=256)),
    "recce":        dict(registry="Recce",           image_size=256, norm="imagenet",
                         init=dict(), need_deepfake_yaml="recce.yaml"),
    "sia":          dict(registry="SIA",             image_size=256, norm="imagenet",
                         init=dict(), need_deepfake_yaml="sia.yaml",
                         deepfake_fallbacks=["SIA", "Sia", "sia"]),
    "iml-vit":      dict(registry="IML_ViT",        image_size=256, norm="imagenet",
                         init=dict(), wrapper="Mask2LabelWrapper",
                         wrapper_init=dict(edge_lambda=20)),
    "trufor":       dict(registry="Trufor",         image_size=256, norm="imagenet",
                         init=dict(), wrapper="Mask2LabelWrapper", wrapper_init=dict()),
    "univfd":       dict(registry="UnivFD",         image_size=224, norm="clip",
                         init=dict(name="ViT-L/14", num_classes=1)),
    "ffdn":         dict(registry="FFDN",           image_size=256, norm="imagenet",
                         init=dict(), wrapper="Mask2LabelWrapper", wrapper_init=dict()),
    "effort":       dict(registry="Effort",         image_size=256, norm="imagenet",
                         init=dict(), wrapper="Mask2LabelWrapper", wrapper_init=dict()),
    "mesorch":      dict(registry="Mesorch",        image_size=256, norm="imagenet",
                         init=dict(), wrapper="Mask2LabelWrapper", wrapper_init=dict()),
    "co-spy":       dict(registry="CO_SPY",         image_size=224, norm="clip",
                         init=dict(clip_model="ViT-L/14", pretrained="datacomp_xl_s13b_b90k", dim_clip=768)),
}

MASK_WRAPPER_NEED = {"IML_ViT", "Mesorch", "Trufor", "Effort", "FFDN", "DTD",
                     "MVSSNet", "PSCC_Net", "Cat_Net"}


def _resolve_deepfake_yaml(name_file):
    """DeepfakeBench detector yamls live inside the installed DeepfakeBench package.

    Robust to namespace-package shadowing: model_factory puts <...>/ForensicHub
    first on sys.path, and on some machines that makes `import DeepfakeBench`
    resolve to a namespace package (no __file__). So probe spec locations
    (which survive shadowing) instead of relying on __file__ alone, descend
    one extra level for checkout-root shadowing, and keep a
    case-insensitive fallback.
    """
    import importlib.util
    roots = []
    try:
        spec = importlib.util.find_spec("DeepfakeBench")
        if spec is not None:
            for loc in (spec.submodule_search_locations or []):
                roots.append(loc)
                # namespace portion may be the repo root; real package is one level deeper
                roots.append(str(Path(loc) / "DeepfakeBench"))
            if spec.origin and spec.origin not in (None, "namespace"):
                roots.append(str(Path(spec.origin).parent))
    except Exception as e:
        warnings.warn(f"DeepfakeBench spec probe failed ({e})")
    try:
        import DeepfakeBench
        f = getattr(DeepfakeBench, "__file__", None)
        if f:
            pp = Path(f).parent
            roots.append(str(pp))
            roots.append(str(pp / "DeepfakeBench"))
    except Exception as e:
        warnings.warn(f"DeepfakeBench yaml resolve failed ({e})")
    subdirs = ("training/config/config/detector", "training/config/detector")
    seen, bases = set(), []
    for r in roots:
        for s in subdirs:
            key = r + "|" + s
            if key not in seen:
                seen.add(key)
                bases.append(Path(r) / s)
    for b in bases:
        try:
            if (b / name_file).exists():
                return str(b / name_file)
        except Exception:
            continue
    for b in bases:
        try:
            for c in b.glob("*.yaml"):
                if c.stem.lower() == Path(name_file).stem.lower():
                    return str(c)
        except Exception:
            continue
    warnings.warn(f"DeepfakeBench yaml \u0027{name_file}\u0027 not found. Tried: {[str(b) for b in bases]}")
    return None


def build_forensic_model(model_key):
    """Return (model, cfg). Raises with actionable message if dep missing."""
    from ForensicHub.registry import MODELS, build_from_registry
    key = model_key.lower()
    if key not in MODEL_CFG:
        raise KeyError(f"Unknown --model {model_key!r}. Choose from: {sorted(MODEL_CFG)}")
    cfg = MODEL_CFG[key]

    # 1) deepfake yaml-based models: fill yaml_config_path automatically
    init = dict(cfg.get("init", {}))
    if cfg.get("need_deepfake_yaml"):
        if not init.get("yaml_config_path"):
            yp = _resolve_deepfake_yaml(cfg["need_deepfake_yaml"])
            if yp is None:
                raise FileNotFoundError(
                    f"[{key}] needs DeepfakeBench config '{cfg['need_deepfake_yaml']}' but "
                    "DeepfakeBench is not installed or the yaml was not found. "
                    "pip install DeepfakeBench (or run: pip install -e <DeepfakeBench>) "
                    "on the remote server first.")
            init["yaml_config_path"] = yp

def _resolve_imdl_asset(*parts):
    """Locate a file inside the installed IMDLBenCo checkout (repo layout varies).

    Probes, in order: the IMDLBenCo package dir, its parent (repo root), and
    the repo-root/model_zoo + statics trees. Returns path str or None.
    """
    try:
        import IMDLBenCo
        pkg = Path(IMDLBenCo.__file__).parent
        cands = [pkg.joinpath(*parts),
                 pkg.parent.joinpath(*parts),
                 pkg.joinpath("model_zoo", *parts),
                 pkg.parent.joinpath("model_zoo", *parts)]
        for c in cands:
            if c.exists():
                return str(c)
    except Exception as e:
        warnings.warn(f"IMDLBenCo asset probe failed ({e})")
    return None

    # 2) FFDN ships a hard-coded weight_path that does not exist remotely -> random init fallback
    if cfg["registry"] == "FFDN":
        try:
            from ForensicHub.tasks.document.models.ffdn.ffdn import FFDN as _FFDN
            _orig = _FFDN.init_vph
            def _safe_init_vph(self, weight_path):
                try:
                    return _orig(self, weight_path)
                except Exception as e:
                    warnings.warn(f"[ffdn] pretrained weights not found ({e}); use random init.")
            _FFDN.init_vph = _safe_init_vph
        except Exception:
            pass


    # 2b) server-side weight paths (resolve dynamically; local/server layouts differ)
    if key == "segformer":
        mit = (_resolve_imdl_asset("model_zoo", "diffiml", "mit_b3.pth")
               or _resolve_imdl_asset("diffiml", "mit_b3.pth"))
        if mit:
            init["pretrain_path"] = mit
        else:
            # mit_b3 nowhere to be found (disk + openmmlab 404): feed an empty
            # state dict so the unconditional torch.load in MixVisionTransformer
            # no-ops under strict=False -> honest RANDOM INIT, same as FFDN.
            import tempfile
            dummy = str(Path(tempfile.gettempdir()) / "segformer_empty_init.pth")
            if not Path(dummy).exists():
                torch.save({}, dummy)
            init["pretrain_path"] = dummy
            warnings.warn("[segformer] mit_b3.pth not found; RANDOM INIT.")
    if key == "trufor":
        wi = dict(cfg.get("wrapper_init", {}))
        np_w = _resolve_imdl_asset("model_zoo", "trufor", "noiseprint.pth")
        mit_w = _resolve_imdl_asset("model_zoo", "trufor", "mit_b2.pth")
        cfg_yaml = (_resolve_imdl_asset("statics", "model_zoo", "configs", "trufor.yaml")
                    or _resolve_imdl_asset("model_zoo", "configs", "trufor.yaml"))
        if np_w:
            wi.setdefault("np_pretrain_weights", np_w)
        if mit_w:
            wi.setdefault("mit_b2_pretrain_weights", mit_w)
        if cfg_yaml:
            wi.setdefault("config_path", cfg_yaml)
        cfg = {**cfg, "wrapper_init": wi}
    def _looks_like_network_err(e):
        s = str(e).lower()
        keys = ["network is unreachable", "huggingface", "client has been closed",
                "connection", "timed out", "max retries", "101",
                "getaddrinfo", "name resolution", "offline"]
        return any(k in s for k in keys)

    # 3) build (optionally wrapped for mask models)
    if cfg.get("wrapper") == "Mask2LabelWrapper" or cfg["registry"] in MASK_WRAPPER_NEED:
        outer = {"name": "Mask2LabelWrapper",
                 "init_config": {"name": cfg["registry"], "init_config": cfg.get("wrapper_init", {})}}
        try:
            model = build_from_registry(MODELS, outer)
        except Exception as e:
            # IMDLBenCo pretrain files (mae_pretrain_vit_base.pth etc.) often missing ->
            # retry with empty init so the model still trains from scratch for diagnosis.
            warnings.warn(f"[{key}] wrapper build failed ({e}); retry without pretrain args.")
            outer["init_config"]["init_config"] = {}
            model = build_from_registry(MODELS, outer)
    else:
        req = {"name": cfg["registry"], "init_config": init}
        # deepfake SIA name may vary in case across DeepfakeBench versions
        names_to_try = [cfg["registry"]] + cfg.get("deepfake_fallbacks", [])
        err = None
        model = None
        for nm in names_to_try:
            try:
                req["name"] = nm
                model = build_from_registry(MODELS, req)
                cfg = {**cfg, "registry": nm}
                break
            except Exception as e:
                err = e
        if model is None and _looks_like_network_err(err) and "pretrained" not in req["init_config"]:
            try:
                warnings.warn(f"[{key}] network blocked, retry with pretrained=False.")
                req2 = {"name": req["name"], "init_config": {**req["init_config"], "pretrained": False}}
                model = build_from_registry(MODELS, req2)
                warnings.warn(f"[{key}] using random init (pretrained=False) for diagnosis.")
            except TypeError as e2:
                # e.g. Segformerb3 takes pretrain_path, not pretrained: keep ORIGINAL error.
                if "pretrained" in str(e2):
                    warnings.warn(f"[{key}] retry not applicable ({e2}); keeping original error.")
                else:
                    err = e2
            except Exception as e2:
                err = e2
        if model is None:
            raise RuntimeError(f"[{key}] build failed. Last error: {err}")
    return model, cfg


class LastLinearInputHook:
    """Capture the input of the last nn.Linear = the linear-probe feature space.

    Works for (almost) all 15 models without knowing their internals:
    the vector that the final linear classifier sees is exactly what a
    linear probe would train on. For Mask2LabelWrapper we hook the inner model.
    """

    def __init__(self, model):
        self.feat = None
        target = model
        # unwrap one level (Mask2LabelWrapper.base_model) for cleaner features
        inner = getattr(model, "base_model", None)
        search_root = inner if isinstance(inner, nn.Module) else model
        self.linear = None
        for m in search_root.modules():
            if isinstance(m, nn.Linear):
                self.linear = m  # keep last one
        if self.linear is None:
            # fallback: last Conv2d (pure mask heads) -> global-average to vector
            for m in search_root.modules():
                if isinstance(m, nn.Conv2d):
                    self.linear = m
        if self.linear is None:
            raise RuntimeError("No Linear/Conv2d head found for feature hook.")
        self.is_conv = isinstance(self.linear, nn.Conv2d)
        self.handle = self.linear.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        x = inp[0].detach()
        if x.dim() == 4:  # [B,C,H,W] -> GAP vector
            x = x.mean(dim=(2, 3))
        self.feat = x

    def close(self):
        try:
            self.handle.remove()
        except Exception:
            pass


@torch.no_grad()
def extract_features(model, loader, device, max_batches=None):
    """Run forward (with dummy aux inputs) and collect hooked features."""
    model.eval()
    hook = LastLinearInputHook(model)
    feats, labels, domains = [], [], []
    try:
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            img = batch["image"].to(device)
            lab = batch["label"].to(device)
            B, _, H, W = img.shape
            aux = dict(
                mask=torch.zeros(B, 1, H, W, device=device),
                edge_mask=torch.zeros(B, 1, H, W, device=device),
                dct=torch.zeros(B, 1, H, W, dtype=torch.long, device=device),
                qt=torch.ones(B, 1, 8, 8, dtype=torch.long, device=device),
                DCT_coef=torch.zeros(B, 1, H, W, dtype=torch.long, device=device),
                qtables=torch.ones(B, 1, 8, 8, dtype=torch.long, device=device),
            )
            try:
                model(image=img, label=lab, **aux)
            except TypeError:
                model(image=img, label=lab)
            if hook.feat is None:
                raise RuntimeError("Feature hook captured nothing; check model forward.")
            feats.append(hook.feat.float().cpu())
            labels.append(batch["label"].cpu())
            domains.append(batch["domain"].cpu())
    finally:
        hook.close()
    import torch as _t
    return {"features": _t.cat(feats).numpy(), "labels": _t.cat(labels).numpy(),
            "domains": _t.cat(domains).numpy()}



