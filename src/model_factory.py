"""Build configured models through the sibling ForensicHub registry.

Sibling layout (local == remote):
  graduation_project/
    ForensicHub/                 <- ForensicHub repo
    linear_probe-and-gradient/   <- this project (you are here)

So ForensicHub is found as <project_root>/../ForensicHub.
"""
import sys
import warnings
import importlib
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[2]          # graduation_project/
FORENSICHUB_REPO = PROJECT_ROOT / "ForensicHub"
for p in (str(FORENSICHUB_REPO),):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from .model_config import MODEL_CFG, get_exclusion_reason
except ImportError:
    from model_config import MODEL_CFG, get_exclusion_reason


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


def _load_task_checkpoint(model, cfg, path):
    """Load a complete detector or its configured task head, with strict matching."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    for field in ("state_dict", "model"):
        if isinstance(state, Mapping) and isinstance(state.get(field), Mapping):
            state = state[field]
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Invalid task checkpoint: {path}")
    if all(name.startswith("module.") for name in state):
        state = {name[len("module."):]: value for name, value in state.items()}
    target_path = cfg.get("task_checkpoint_target", "")
    target = _resolve_module_path(model, target_path) if target_path else model
    prefix = target_path + "."
    if target_path and all(name.startswith(prefix) for name in state):
        state = {name[len(prefix):]: value for name, value in state.items()}
    target.load_state_dict(state, strict=True)
    print(f"[diag] loaded task checkpoint={path} "
          f"target={target_path or 'model'} tensors={len(state)} strict=True", flush=True)


def _build_task_deepfake(registry_name, yaml_config_path):
    """Build the native detector without a redundant ImageNet weight download."""
    from DeepfakeBench.training import detectors
    from DeepfakeBench.training.networks import BACKBONE
    from ForensicHub.tasks.deepfake.wrapper.wrappers import Deepfake2ForensicWrapper

    detector_class = getattr(detectors, registry_name)

    class TaskDetector(detector_class):
        def build_backbone(self, config):
            backbone_config = dict(config["backbone_config"])
            if config["model_name"] == "spsl":
                backbone_config["inc"] = 4
            elif config["model_name"] == "f3net":
                backbone_config["inc"] = 12
            return BACKBONE[config["backbone_name"]](backbone_config)

    return Deepfake2ForensicWrapper(TaskDetector, yaml_config_path)


class LabelOnlyMaskWrapper(nn.Module):
    """Train a native localizer using the project's image labels only.

    Pixel losses computed with placeholder masks are discarded; only image BCE
    updates the model. Per-model settings select a segmentation branch when the
    native image head detaches the backbone, or handle raw logits explicitly.
    """

    def __init__(self, base_model, prediction_field=None, logits=False):
        super().__init__()
        if prediction_field not in (None, "pred_label", "pred_mask"):
            raise ValueError(f"Unknown label prediction field: {prediction_field}")
        self.base_model = base_model
        self.prediction_field = prediction_field
        self.logits = logits

    def forward(self, image, label, **kwargs):
        batch, _, height, width = image.shape
        kwargs.setdefault("mask", image.new_zeros(batch, 1, height, width))
        kwargs.setdefault("edge_mask", image.new_zeros(batch, 1, height, width))
        result = self.base_model(image=image, label=label, **kwargs)
        pred = result.get("pred_label") if self.prediction_field != "pred_mask" else None
        if pred is None:
            mask = result.get("pred_mask")
            if mask is None or mask.ndim != 4:
                raise ValueError("Localizer must return pred_label or [B,C,H,W] pred_mask")
            pred = mask[:, -1:].amax(dim=(1, 2, 3))
        pred = pred.reshape(-1)
        if pred.numel() != label.numel():
            raise ValueError("Localizer image predictions do not match batch labels")
        target = label.float().reshape(-1)
        if self.logits:
            loss = nn.functional.binary_cross_entropy_with_logits(pred, target)
            pred = pred.sigmoid()
        else:
            loss = nn.functional.binary_cross_entropy(pred, target)
        return {"backward_loss": loss, "pred_label": pred,
                "visual_loss": {"image_label_loss": loss}}


def build_forensic_model(model_key, task_checkpoint=None, task_checkpoint_target=None,
                         require_task_checkpoint=False, device=None):
    """Return (model, cfg). Raises with actionable message if dep missing."""
    key = model_key.lower()
    exclusion = get_exclusion_reason(model_key)
    if exclusion:
        raise ValueError(f"[{model_key}] outside experiment scope: {exclusion}")
    if key not in MODEL_CFG:
        raise KeyError(f"Unknown --model {model_key!r}. Choose from: {sorted(MODEL_CFG)}")
    cfg = dict(MODEL_CFG[key])
    if cfg.get("blocked_reason"):
        raise RuntimeError(f"[{key}] project adapter unavailable: {cfg['blocked_reason']}")
    if task_checkpoint_target is not None:
        cfg["task_checkpoint_target"] = task_checkpoint_target
    task_checkpoint = task_checkpoint or cfg.get("task_checkpoint")
    if require_task_checkpoint and not task_checkpoint:
        raise ValueError(f"[{key}] --require_task_checkpoint needs a detector checkpoint; "
                         "backbone initialization alone does not qualify")
    if task_checkpoint:
        task_checkpoint = Path(task_checkpoint).expanduser()
        if not task_checkpoint.is_absolute():
            task_checkpoint = THIS.parents[1] / task_checkpoint
        task_checkpoint = task_checkpoint.resolve()
        if not task_checkpoint.is_file():
            raise FileNotFoundError(f"[{key}] task checkpoint not found: {task_checkpoint}")
        if cfg.get("task_norm"):
            cfg = {**cfg, "norm": cfg["task_norm"]}
        cfg["task_checkpoint"] = str(task_checkpoint)

    from ForensicHub.registry import MODELS, build_from_registry
    if cfg.get("import_module"):
        importlib.import_module(cfg["import_module"])

    for field in ("init", "wrapper_init"):
        resolved = dict(cfg.get(field, {}))
        if device is not None and "device" in resolved:
            resolved["device"] = str(device)
        for name, value in resolved.items():
            if isinstance(value, str) and name.endswith(("_path", "_weights")):
                asset = Path(value).expanduser()
                if not asset.is_absolute():
                    asset = THIS.parents[1] / asset
                resolved[name] = str(asset.resolve())
        if field in cfg:
            cfg[field] = resolved

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
    if key == "segformer" and not init.get("pretrain_path"):
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
    if (task_checkpoint and cfg.get("task_checkpoint_target") == "_base_model"
            and cfg["registry"] in {"SpslDetector", "SRMDetector", "F3netDetector", "XceptionDetector"}):
        model = _build_task_deepfake(cfg["registry"], init["yaml_config_path"])
    elif cfg.get("wrapper") == "LabelOnlyMaskWrapper":
        model = LabelOnlyMaskWrapper(build_from_registry(MODELS, {
            "name": cfg["registry"], "init_config": cfg.get("wrapper_init", init)}),
            **cfg.get("label_adapter", {}))
    elif cfg.get("wrapper") == "Mask2LabelWrapper":
        outer = {"name": "Mask2LabelWrapper",
                 "init_config": {"name": cfg["registry"], "init_config": cfg.get("wrapper_init", {})}}
        try:
            model = build_from_registry(MODELS, outer)
        except Exception as e:
            if task_checkpoint:
                raise
            # IMDLBenCo pretrain files (mae_pretrain_vit_base.pth etc.) often missing ->
            # retry with empty init so the model still trains from scratch for diagnosis.
            warnings.warn(f"[{key}] wrapper build failed ({e}); retry without pretrain args.")
            outer["init_config"]["init_config"] = {}
            model = build_from_registry(MODELS, outer)
    else:
        req = {"name": cfg["registry"], "init_config": init}
        err = None
        model = None
        try:
            model = build_from_registry(MODELS, req)
        except Exception as e:
            err = e
        if (model is None and not task_checkpoint and _looks_like_network_err(err)
                and "pretrained" not in req["init_config"]):
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
    if task_checkpoint:
        _load_task_checkpoint(model, cfg, task_checkpoint)
    return model, cfg


FEATURE_TARGETS = {key: cfg["feature"] for key, cfg in MODEL_CFG.items() if "feature" in cfg}


def _unwrap_feature_root(model):
    """Peel DDP and the local zero-anchor wrapper before resolving hook paths."""
    root = model
    seen = set()
    while id(root) not in seen:
        seen.add(id(root))
        if isinstance(root, torch.nn.parallel.DistributedDataParallel):
            root = root.module
            continue
        if root.__class__.__name__ == "DDPForwardAnchor":
            inner = getattr(root, "model", None)
            if isinstance(inner, nn.Module):
                root = inner
                continue
        break
    return root


def _resolve_module_path(root, dotted_path):
    obj = root
    for part in dotted_path.split("."):
        if not hasattr(obj, part):
            raise AttributeError(
                f"Feature hook target {dotted_path!r} not found: "
                f"{type(obj).__name__} has no attribute {part!r}"
            )
        obj = getattr(obj, part)
    if not isinstance(obj, nn.Module):
        raise TypeError(
            f"Feature hook target {dotted_path!r} resolved to "
            f"{type(obj).__name__}, not nn.Module"
        )
    return obj


class RepresentationHook:
    """Capture a comparable semantic representation for rank diagnostics.

    Most models use the input of the last Linear/Conv classifier, preserving the
    original behaviour.  IML-ViT and TruFor need explicit semantic bottlenecks:
    their last Linear layers are task-specific heads and are much narrower than
    the encoder representation.  For those models, hook the output of the
    configured fusion layer and globally pool it to a vector.
    """

    def __init__(self, model, model_key=None):
        self.feat = None
        self.model_key = model_key.lower() if model_key else None
        self.expected_dim = None
        model = _unwrap_feature_root(model)
        target = FEATURE_TARGETS.get(self.model_key)
        if target:
            self.target_path = target["path"]
            self.expected_dim = target.get("expected_dim")
            self.mode = target.get("source", "output")
            module = _resolve_module_path(model, self.target_path)
            self.description = (
                f"{self.model_key}:{self.mode}:{self.target_path}:"
                f"expected_dim={self.expected_dim}"
            )
            self.handle = module.register_forward_hook(
                self._hook_input if self.mode == "input" else self._hook_output
            )
            return

        # Generic fallback: the vector entering the final linear classifier.
        inner = getattr(model, "base_model", None)
        search_root = inner if isinstance(inner, nn.Module) else model
        self.linear = None
        self.linear_name = None
        for name, module in search_root.named_modules():
            if isinstance(module, nn.Linear):
                self.linear = module
                self.linear_name = name
        if self.linear is None:
            for name, module in search_root.named_modules():
                if isinstance(module, nn.Conv2d):
                    self.linear = module
                    self.linear_name = name
        if self.linear is None:
            raise RuntimeError("No Linear/Conv2d head found for feature hook.")
        self.mode = "input"
        self.target_path = self.linear_name or type(self.linear).__name__
        self.description = f"{self.model_key or 'unknown'}:{self.mode}:{self.target_path}"
        self.handle = self.linear.register_forward_hook(self._hook_input)

    @staticmethod
    def _pool(value):
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not torch.is_tensor(value):
            raise TypeError(f"Feature hook expected a tensor, got {type(value).__name__}")
        value = value.detach()
        if value.dim() == 4:       # [B,C,H,W] -> [B,C]
            value = value.mean(dim=(2, 3))
        elif value.dim() == 3:     # [B,N,C] -> [B,C]
            value = value.mean(dim=1)
        elif value.dim() > 2:      # conservative flatten for unusual heads
            value = value.flatten(1)
        return value

    def _assign_feat(self, value):
        feat = self._pool(value)
        if self.expected_dim is not None and feat.shape[-1] != self.expected_dim:
            raise RuntimeError(
                f"Feature hook {self.target_path!r} produced dim={feat.shape[-1]}, "
                f"expected {self.expected_dim}"
            )
        self.feat = feat

    def _hook_input(self, module, inp, out):
        self._assign_feat(inp[0])

    def _hook_output(self, module, inp, out):
        self._assign_feat(out)

    def close(self):
        try:
            self.handle.remove()
        except Exception:
            pass


# Backward-compatible name for notebooks/scripts that imported the old class.
LastLinearInputHook = RepresentationHook


@torch.no_grad()
def extract_features(model, loader, device, max_batches=None, model_key=None):
    """Run forward (with dummy aux inputs) and collect hooked features."""
    model.eval()
    hook = RepresentationHook(model, model_key=model_key)
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
            "domains": _t.cat(domains).numpy(), "feature_hook": hook.description}



