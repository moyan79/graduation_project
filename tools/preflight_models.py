"""Check checkpoint loading, image loss, gradients and features before a full run."""
import argparse
import gc
import json
from pathlib import Path
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def check_model(key, args, torch):
    from model_factory import build_forensic_model, extract_features
    import numpy as np

    torch.manual_seed(42)
    model, config = build_forensic_model(
        key, task_checkpoint=args.task_checkpoint,
        task_checkpoint_target=args.task_checkpoint_target,
        require_task_checkpoint=args.require_task_checkpoint, device=args.device)
    model.to(args.device).train()
    size = args.image_size or config["image_size"]
    images = torch.randn(args.batch_size, 3, size, size, device=args.device)
    labels = (torch.arange(args.batch_size, device=args.device) % 2).float()
    aux = {"mask": images.new_zeros(args.batch_size, 1, size, size),
           "edge_mask": images.new_zeros(args.batch_size, 1, size, size),
           "dct": torch.zeros(args.batch_size, 1, size, size, dtype=torch.long, device=args.device),
           "qt": torch.ones(args.batch_size, 1, 8, 8, dtype=torch.long, device=args.device)}
    aux.update(DCT_coef=aux["dct"], qtables=aux["qt"])
    output = model(image=images, label=labels, **aux)
    loss = output["backward_loss"].mean()
    if not loss.requires_grad or not torch.isfinite(loss):
        raise RuntimeError("Image loss must be finite and differentiable")
    prediction = output["pred_label"]
    if prediction.numel() != args.batch_size or not torch.isfinite(prediction).all():
        raise RuntimeError("Image predictions must be finite and match the batch size")
    loss.backward()
    gradients = [param.grad for param in model.parameters() if param.grad is not None]
    if not gradients or not all(torch.isfinite(grad).all() for grad in gradients):
        raise RuntimeError("Parameter gradients are missing or non-finite")
    if not any(torch.count_nonzero(grad).item() for grad in gradients):
        raise RuntimeError("All parameter gradients are zero")
    gradient_count = len(gradients)
    model.zero_grad(set_to_none=True)
    batch = {"image": images.cpu(), "label": labels.cpu(),
             "domain": torch.arange(args.batch_size) % 4}
    features = extract_features(model, [batch], args.device, model_key=key)
    if features["features"].shape[0] != args.batch_size or not np.isfinite(features["features"]).all():
        raise RuntimeError("Diagnostic features must be finite and match the batch size")
    return {"status": "passed", "registry": config["registry"], "image_size": size,
            "loss": float(loss.detach()), "gradient_parameter_count": gradient_count,
            "feature_shape": list(features["features"].shape), "feature_hook": features["feature_hook"],
            "task_checkpoint": config.get("task_checkpoint")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--task_checkpoint")
    parser.add_argument("--task_checkpoint_target")
    parser.add_argument("--require_task_checkpoint", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--online", action="store_true", help="explicitly allow missing initialization downloads")
    parser.add_argument("--report", type=Path, default=PROJECT / "reports" / "model_preflight.json")
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("--batch_size must be at least 2 for normalization and both labels")
    if args.task_checkpoint and len(args.model) != 1:
        parser.error("a checkpoint override requires exactly one model")
    import offline_patch
    offline_patch.apply(online=args.online)
    import torch

    torch.set_num_threads(2)
    results = []
    for key in args.model:
        row = {"model": key, "status": "failed", "synthetic_preflight_only": True}
        try:
            row.update(check_model(key, args, torch))
        except Exception as error:
            row.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(row)
        print(f"[preflight] {key}: {row['status']} {row.get('error', row.get('feature_shape'))}", flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"scope": "synthetic build/loss/gradient/feature checks; not completed experiments",
                                          "device": args.device, "results": results},
                                         ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if any(row["status"] != "passed" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
