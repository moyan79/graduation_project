import csv, json, math, re
from pathlib import Path

BASE = Path(r"D:\GPT\gpt\graduation_project\linear_probe-and-gradient\outputs")
OUT = Path(r"D:\GPT\gpt\graduation_project\linear_probe-and-gradient\_results.json")

DISPLAY = {
    "diag_ddp_resnet": ("ResNet101", "ResNet101"),
    "diag_ddp_efficientnet": ("EfficientNet-B4", "EfficientB4"),
    "diag_ddp_capsulenet": ("CapsuleNet", "Capsule_net"),
    "diag_ddp_segformer": ("SegFormer-B3", "Segformerb3"),
    "diag_ddp_swin": ("Swin-Small", "SwinSmall"),
    "diag_ddp_convnext": ("ConvNeXt-Small", "ConvNextSmall"),
    "diag_ddp_recce": ("RECCE", "Recce"),
    "diag_ddp_iml-vit": ("IML-ViT", "IML_ViT"),
    "diag_ddp_trufor": ("TruFor", "Trufor"),
    "diag_ddp_univfd": ("UnivFD", "UnivFD"),
    "diag_ddp_mesorch": ("Mesorch", "Mesorch"),
    "diag_ddp_co-spy": ("CO-SPY", "CO_SPY"),
    "diag_ddp_ffdn": ("FFDN", "FFDN"),
}

INIT = {
    "diag_ddp_resnet": "timm ImageNet 预训练",
    "diag_ddp_efficientnet": "timm ImageNet 预训练",
    "diag_ddp_capsulenet": "VGG19 预训练 + DeepfakeBench 配置",
    "diag_ddp_segformer": "随机初始化（mit_b3 权重缺失，空 state_dict 兜底）",
    "diag_ddp_swin": "timm ImageNet 预训练",
    "diag_ddp_convnext": "timm ImageNet 预训练",
    "diag_ddp_recce": "Xception 预训练权重 + DeepfakeBench 配置",
    "diag_ddp_iml-vit": "IMDLBenCo 模型默认初始化/可用预训练权重",
    "diag_ddp_trufor": "Noiseprint + MiT-B2 预训练权重 + TruFor 配置",
    "diag_ddp_univfd": "CLIP ViT-L/14 预训练",
    "diag_ddp_mesorch": "IMDLBenCo 模型默认初始化/可用预训练权重",
    "diag_ddp_co-spy": "CLIP 与 datacomp 预训练权重 + Stable Diffusion 组件",
    "diag_ddp_ffdn": "FFDN 默认 ConvNeXt-Small 权重路径（日志无加载失败）",
}

NOTE = {
    "diag_ddp_resnet": "基准模型；预训练版结果",
    "diag_ddp_efficientnet": "有效秩较高，谱分布相对分散",
    "diag_ddp_capsulenet": "钩子特征仅 64 维，跨模型比较需谨慎",
    "diag_ddp_segformer": "随机初始化，结果不能视作预训练基线",
    "diag_ddp_swin": "平均梯度余弦为正，冲突程度较低",
    "diag_ddp_convnext": "梯度冲突率处于较低水平",
    "diag_ddp_recce": "钩子输出 16×512，展平为 8192 维",
    "diag_ddp_iml-vit": "钩子仅捕获 16 维小头，秩指标不作为有效比较",
    "diag_ddp_trufor": "128 维且有效秩约 3，疑似钩子落在退化小头",
    "diag_ddp_univfd": "cumvar@20 最低、有效秩最高，谱分布最分散",
    "diag_ddp_mesorch": "钩子输出 64×2048，展平为 131072 维",
    "diag_ddp_co-spy": "参数量约 512.7M；日志实际单卡 batch 为 32",
    "diag_ddp_ffdn": "512 输入、单卡 batch 16，首个反传 OOM",
}

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

def f(x):
    try:
        return float(str(x).replace("+", ""))
    except Exception:
        return None

def read_loss(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            out = {"epoch": int(float(row["epoch"]))}
            for k in ("steps",):
                try: out[k] = int(float(row[k]))
                except Exception: out[k] = None
            for k in ("mean_train_loss", "mean_conflict_ratio", "mean_cos"):
                out[k] = f(row.get(k))
            rows.append(out)
    return rows

records = []
for d in sorted(p for p in BASE.iterdir() if p.is_dir()):
    name = d.name
    display, registry = DISPLAY.get(name, (name, name))
    r = read_json(d / "rank_report.json")
    g = read_json(d / "gradient_report.json")
    losses = read_loss(d / "epoch_loss.csv")
    log = (d / "train.log").read_text(encoding="utf-8", errors="replace") if (d / "train.log").exists() else ""

    m = re.search(r"\[diag\] model=.*?world=(\d+) per_gpu_bs=(\d+) \(global_bs=(\d+)\)", log)
    world, per_gpu, global_bs = (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (None, None, None)
    b = re.search(r"\[diag\] built .*?params=([0-9.]+)M img=(\d+) norm=(\w+)", log)
    params_m, image_size, norm = (float(b.group(1)), int(b.group(2)), b.group(3)) if b else (None, None, None)
    shape = re.search(r"\[diag\] dumped features_epoch0\.npz \((.*?)\)", log)

    traj = []
    tdir = d / "traj"
    if tdir.exists():
        for sub in sorted(tdir.glob("features_epoch*"), key=lambda p: int(re.search(r"(\d+)$", p.name).group(1))):
            tr = read_json(sub / "rank_report.json")
            if tr:
                traj.append({"epoch": int(re.search(r"(\d+)$", sub.name).group(1)), "rank": tr})

    loss_first = losses[0]["mean_train_loss"] if losses else None
    loss_last = losses[-1]["mean_train_loss"] if losses else None
    loss_min = min((x["mean_train_loss"] for x in losses if x["mean_train_loss"] is not None), default=None)
    loss_change_pct = ((loss_last - loss_first) / abs(loss_first) * 100.0) if loss_first not in (None, 0) and loss_last is not None else None

    rank_metric = None
    rank_metric_name = None
    if r:
        if "cumvar_top20" in r:
            rank_metric = r.get("cumvar_top20"); rank_metric_name = "cumvar@20"
        elif "cumvar_top16" in r:
            rank_metric = r.get("cumvar_top16"); rank_metric_name = "cumvar@16"

    image_files = {}
    for fn in ("svd_cumsum.png", "svd_spectrum.png", "grad_conflict_curve.png", "grad_cos_bar.png"):
        if (d / fn).exists(): image_files[fn] = str(d / fn)

    rec = {
        "dir": name,
        "model": display,
        "registry": registry,
        "status": "完成" if r and g and losses else "失败/未完成",
        "init": INIT.get(name, "未记录"),
        "note": NOTE.get(name, ""),
        "world": world,
        "per_gpu_bs": per_gpu,
        "global_bs": global_bs,
        "params_m": params_m,
        "image_size": image_size,
        "norm": norm,
        "feature_shape": shape.group(1) if shape else None,
        "rank": r,
        "rank_metric": rank_metric,
        "rank_metric_name": rank_metric_name,
        "gradient": g,
        "losses": losses,
        "loss_first": loss_first,
        "loss_last": loss_last,
        "loss_min": loss_min,
        "loss_change_pct": loss_change_pct,
        "traj_first": traj[0]["rank"] if traj else None,
        "traj_last": traj[-1]["rank"] if traj else None,
        "traj_count": len(traj),
        "images": image_files,
    }
    records.append(rec)

OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
for x in records:
    r=x["rank"]
    g=x["gradient"]
    print("|".join(str(v) for v in [
        x["model"], x["status"], x["params_m"], x["image_size"], f"{x['per_gpu_bs']}/{x['global_bs']}",
        r.get("dim") if r else None,
        round(x["rank_metric"],4) if x["rank_metric"] is not None else None,
        round(r.get("effective_rank"),2) if r and r.get("effective_rank") is not None else None,
        round(g.get("mean_conflict_ratio"),4) if g and g.get("mean_conflict_ratio") is not None else None,
        round(g.get("mean_cos_overall"),4) if g and g.get("mean_cos_overall") is not None else None,
        round(x["loss_first"],4) if x["loss_first"] is not None else None,
        round(x["loss_last"],4) if x["loss_last"] is not None else None,
    ]))
print("WROTE", OUT)
