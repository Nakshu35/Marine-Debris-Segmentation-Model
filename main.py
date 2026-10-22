import argparse
import json
import os
import shutil
import time
from copy import deepcopy

import torch
import yaml

from data.marida_dataset import CLASS_NAMES, IGNORE_INDEX, MARIDADataset, NUM_CLASSES
from data.ssl_dataset import IndianCoastalSSLDataset
from data.transforms import NUM_BANDS
from evaluation.confusion_matrix import save_confusion_matrix_csv
from evaluation.metrics import SegmentationMetrics
from losses.segmentation_losses import build_loss, compute_class_weights
from models.encoder import ViTEncoder, attach_band_adapter, build_encoder, load_external_checkpoint
from models.segmentation_model import SegmentationModel
from models.ssl_model import MaskedAutoencoderSSL
from training.checkpoint import load_checkpoint
from training.segmentation_trainer import SegmentationTrainer
from training.ssl_trainer import SSLTrainer
from utils.logger import build_console_logger
from utils.seed import set_seed
from utils.system import collect_system_info, get_device
from utils.visualization import save_confusion_matrix_figure, save_prediction_overlay, save_sanity_figure, s2_to_rgb

EXPERIMENTS = ("baseline", "pretrained", "ssl_adapted", "ssl_from_scratch")
MODES = ("sanity", "ssl", "segmentation", "test", "full")


def parse_args():
    p = argparse.ArgumentParser(description="Marine Sentinel-2 SSL + U-Net++ segmentation pipeline")
    p.add_argument("--Config", type=str, default="config/Config.yaml")
    p.add_argument("--mode", type=str, default="full", choices=MODES)
    p.add_argument("--experiment", type=str, default="ssl_adapted", choices=EXPERIMENTS)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file '{path}' did not contain a YAML mapping.")
    return cfg


def _resolve_output_root(cfg: dict) -> str:
    configured = cfg["output"]["runs_dir"]
    if os.path.isabs(configured):
        output_root = configured
    elif os.path.isdir("/kaggle/working"):
        output_root = os.path.join("/kaggle/working", configured)
    else:
        output_root = os.path.abspath(configured)
    output_root = os.path.abspath(output_root)
    cfg["output"]["runs_dir"] = output_root
    return output_root


def setup_run_dir(cfg: dict, experiment: str, config_path: str) -> str:
    output_root = _resolve_output_root(cfg)
    run_dir = os.path.join(output_root, experiment)
    for sub in ("checkpoints", "logs", "predictions", "visualizations"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    shutil.copy2(config_path, os.path.join(run_dir, "config.yaml"))
    return run_dir


def save_run_metadata(run_dir: str, cfg: dict, experiment: str, device, dataset_info: dict) -> None:
    meta = {
        "experiment_name": experiment,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": cfg["seed"],
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "ignore_index": IGNORE_INDEX,
        "model_config": cfg["model"],
        "ssl_config": cfg["ssl"],
        "segmentation_config": cfg["segmentation"],
        "loss_config": cfg["loss"],
        "dataset_info": dataset_info,
        "system_info": collect_system_info(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


class CheckpointSafeDataParallel(torch.nn.DataParallel):
    """DataParallel wrapper that saves/loads without the 'module.' key prefix."""

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        cleaned = _strip_module_prefix_if_needed(state_dict)
        return self.module.load_state_dict(cleaned, *args, **kwargs)


def _strip_module_prefix_if_needed(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if keys and all(isinstance(k, str) and k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def setup_parallel_model(model, device, logger=None):
    model = model.to(device)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        msg = f"Using {gpu_count} CUDA GPUs with DataParallel."
        logger.info(msg) if logger else print(msg)
        return CheckpointSafeDataParallel(model)
    msg = f"Using single device: {device}"
    logger.info(msg) if logger else print(msg)
    return model


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def log_gpu_environment(logger):
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Running on CPU.")
        return
    count = torch.cuda.device_count()
    logger.info(f"CUDA available. Visible GPU count: {count}")
    for i in range(count):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


def load_model_checkpoint_compatible(path: str, model, device, logger=None):
    try:
        return load_checkpoint(path, model, map_location=str(device))
    except (RuntimeError, KeyError, ValueError) as first_error:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            candidate = None
            for key in ("model", "state_dict", "model_state_dict", "model_state"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    candidate = value
                    break
            if candidate is None and checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
                candidate = checkpoint
            if candidate is not None:
                cleaned = _strip_module_prefix_if_needed(candidate)
                result = model.load_state_dict(cleaned, strict=False)
                missing = list(getattr(result, "missing_keys", []))
                unexpected = list(getattr(result, "unexpected_keys", []))
                if logger:
                    logger.warning(
                        f"Compatibility load for '{path}'. "
                        f"Missing: {len(missing)}, unexpected: {len(unexpected)}."
                    )
                if not missing and not unexpected:
                    return checkpoint
        raise first_error


def build_experiment_encoder(cfg: dict, experiment: str, logger) -> ViTEncoder:
    encoder = build_encoder(cfg["model"])

    if experiment == "baseline":
        logger.info("[baseline] Random encoder init.")
        return encoder

    ckpt_path = cfg["model"].get("pretrained_checkpoint", "")
    if experiment in ("pretrained", "ssl_adapted"):
        if ckpt_path:
            ckpt_bands = cfg["model"].get("pretrained_checkpoint_band_count", NUM_BANDS)
            if ckpt_bands != NUM_BANDS:
                logger.info(f"[{experiment}] Checkpoint has {ckpt_bands} bands; attaching BandAdapter.")
                attach_band_adapter(encoder, ckpt_bands)
            load_external_checkpoint(encoder, ckpt_path)
        else:
            logger.info(f"[{experiment}] No pretrained_checkpoint configured; using random init.")
        return encoder

    if experiment == "ssl_from_scratch":
        logger.info("[ssl_from_scratch] Random encoder init; SSL trains from scratch.")
        return encoder

    raise ValueError(f"Unknown experiment '{experiment}'")


def experiment_needs_ssl(experiment: str) -> bool:
    return experiment in ("ssl_adapted", "ssl_from_scratch")


def run_sanity(cfg: dict, device, logger) -> None:
    logger.info("SANITY MODE")

    ssl_ds = IndianCoastalSSLDataset(cfg["data"]["ssl_root"], crop_size=cfg["data"]["ssl_crop_size"])
    marida_ds = MARIDADataset(cfg["data"]["marida_root"], split="train", crop_size=cfg["data"]["seg_crop_size"])

    ssl_view_a, ssl_view_b = ssl_ds[0]
    seg_img, seg_mask = marida_ds[0]

    logger.info(f"SSL view: {tuple(ssl_view_a.shape)}  min={ssl_view_a.min():.4f} max={ssl_view_a.max():.4f}")
    logger.info(f"MARIDA image: {tuple(seg_img.shape)}  min={seg_img.min():.4f} max={seg_img.max():.4f}")

    if seg_img.max().item() == 0.0 and seg_img.min().item() == 0.0:
        logger.warning("MARIDA sample is ALL ZEROS — check marida_root path or .tif file.")

    uniq_labels = torch.unique(seg_mask).tolist()
    logger.info(f"MARIDA mask: {tuple(seg_mask.shape)}  unique ids: {uniq_labels}")
    bad_ids = [i for i in uniq_labels if i != IGNORE_INDEX and not (0 <= i < NUM_CLASSES)]
    assert not bad_ids, f"Invalid label ids: {bad_ids}"
    logger.info("Label ids OK.")

    encoder = build_encoder(cfg["model"]).to(device)

    ssl_model = MaskedAutoencoderSSL(
        encoder,
        decoder_embed_dim=cfg["ssl"]["decoder_embed_dim"],
        decoder_depth=cfg["ssl"]["decoder_depth"],
        decoder_num_heads=cfg["ssl"]["decoder_num_heads"],
        mask_ratio=cfg["ssl"]["mask_ratio"],
        norm_pix_loss=cfg["ssl"]["norm_pix_loss"],
    ).to(device)

    batch_ssl = torch.stack([ssl_view_a, ssl_view_b], dim=0).to(device)
    ssl_loss, pred, mask = ssl_model(batch_ssl)
    logger.info(f"SSL forward OK. loss={ssl_loss.item():.5f} pred={tuple(pred.shape)}")
    assert torch.isfinite(ssl_loss), "SSL loss is NaN/Inf!"

    seg_model = SegmentationModel(
        encoder, num_classes=NUM_CLASSES,
        pyramid_channels=cfg["model"]["pyramid_channels"],
        deep_supervision=cfg["model"]["deep_supervision"],
    ).to(device)

    batch_img = seg_img.unsqueeze(0).to(device)
    batch_mask = seg_mask.unsqueeze(0).to(device)
    logits = seg_model(batch_img)
    logger.info(f"Seg forward OK (MARIDA). logits={tuple(logits.shape)}")
    assert logits.shape == (1, NUM_CLASSES, batch_img.shape[-2], batch_img.shape[-1])

    loss_fn = build_loss(cfg["loss"])
    loss = loss_fn(logits, batch_mask)
    logger.info(f"Loss ({cfg['loss']['mode']}) = {loss.item():.5f} (MARIDA)")
    seg_model.zero_grad()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in seg_model.parameters() if p.grad is not None]
    assert grad_norms, "No gradients computed!"
    nan_grads = [n for n, p in seg_model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not nan_grads, f"NaN/Inf gradients in: {nan_grads}"
    logger.info("Backward OK (MARIDA).")

    vis_dir = os.path.join(cfg["output"]["runs_dir"], "sanity_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    save_sanity_figure(s2_to_rgb(ssl_view_a.numpy()), "SSL sample", os.path.join(vis_dir, "sanity_ssl_sample.png"))
    save_sanity_figure(s2_to_rgb(seg_img.numpy()), "MARIDA sample", os.path.join(vis_dir, "sanity_marida_sample.png"), mask=seg_mask.numpy())
    logger.info(f"Sanity visualizations saved to '{vis_dir}'.")
    logger.info("SANITY PASSED.")


def run_ssl_stage(cfg: dict, experiment: str, encoder: ViTEncoder, device, run_dir: str, resume: bool, logger) -> str:
    logger.info(f"SSL STAGE (experiment={experiment})")

    ssl_dataset = IndianCoastalSSLDataset(cfg["data"]["ssl_root"], crop_size=cfg["data"]["ssl_crop_size"])
    stats = ssl_dataset.get_stats()
    logger.info(f"SSL dataset: {stats['num_files']} patches.")

    ssl_model = MaskedAutoencoderSSL(
        encoder,
        decoder_embed_dim=cfg["ssl"]["decoder_embed_dim"],
        decoder_depth=cfg["ssl"]["decoder_depth"],
        decoder_num_heads=cfg["ssl"]["decoder_num_heads"],
        mask_ratio=cfg["ssl"]["mask_ratio"],
        norm_pix_loss=cfg["ssl"]["norm_pix_loss"],
    )
    ssl_model = setup_parallel_model(ssl_model, device, logger)

    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 1
    effective_batch = int(cfg["ssl"]["batch_size"])
    if gpu_count > 1 and effective_batch < gpu_count:
        logger.warning(f"SSL batch_size={effective_batch} < GPU count={gpu_count}.")

    ssl_cfg = deepcopy(cfg["ssl"])
    ssl_cfg["seed"] = cfg["seed"]
    trainer = SSLTrainer(ssl_model, ssl_dataset, ssl_cfg, device, run_dir, val_fraction=cfg["ssl"]["val_fraction"])

    last_ckpt = os.path.join(run_dir, "checkpoints", "ssl_last.pth")
    if resume and os.path.exists(last_ckpt):
        trainer.resume(last_ckpt)

    best_ckpt_path = trainer.fit()
    logger.info(f"SSL done. Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path


def build_segmentation_model(cfg: dict, encoder: ViTEncoder) -> SegmentationModel:
    return SegmentationModel(
        encoder, num_classes=NUM_CLASSES,
        pyramid_channels=cfg["model"]["pyramid_channels"],
        deep_supervision=cfg["model"]["deep_supervision"],
    )


def run_segmentation_stage(cfg: dict, experiment: str, encoder: ViTEncoder, device, run_dir: str, resume: bool, logger) -> str:
    logger.info(f"SEGMENTATION STAGE (experiment={experiment})")

    train_ds = MARIDADataset(cfg["data"]["marida_root"], split="train", crop_size=cfg["data"]["seg_crop_size"])
    val_ds = MARIDADataset(cfg["data"]["marida_root"], split="val", crop_size=cfg["data"]["seg_crop_size"])
    train_ds.print_statistics()
    val_ds.print_statistics()

    train_counts = train_ds.compute_class_pixel_counts()
    class_weights = compute_class_weights(
        torch.tensor(train_counts), method=cfg["loss"]["class_weight_method"]
    ).to(device)
    logger.info(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")

    loss_fn = build_loss(deepcopy(cfg["loss"]), class_weights=class_weights)
    model = build_segmentation_model(cfg, encoder)
    model = setup_parallel_model(model, device, logger)

    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 1
    effective_batch = int(cfg["segmentation"]["batch_size"])
    if gpu_count > 1 and effective_batch < gpu_count:
        logger.warning(f"Segmentation batch_size={effective_batch} < GPU count={gpu_count}.")

    trainer = SegmentationTrainer(model, train_ds, val_ds, loss_fn, deepcopy(cfg["segmentation"]), device, run_dir)

    last_ckpt = os.path.join(run_dir, "checkpoints", "segmentation_last.pth")
    if resume and os.path.exists(last_ckpt):
        trainer.resume(last_ckpt)

    best_ckpt_path = trainer.fit()
    logger.info(f"Segmentation done. Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path


def run_test_stage(cfg: dict, experiment: str, device, run_dir: str, logger) -> dict:
    logger.info(f"TEST STAGE (experiment={experiment})")

    best_ckpt_path = os.path.join(run_dir, "checkpoints", "segmentation_best.pth")
    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"No checkpoint at '{best_ckpt_path}'. Run segmentation stage first.")

    encoder = build_encoder(cfg["model"])
    model = build_segmentation_model(cfg, encoder).to(device)
    load_model_checkpoint_compatible(best_ckpt_path, model, device, logger)
    model.eval()

    test_ds = MARIDADataset(cfg["data"]["marida_root"], split="test", crop_size=cfg["data"]["seg_crop_size"])
    test_ds.print_statistics()
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg["segmentation"].get("val_batch_size", 16),
        shuffle=False, num_workers=cfg["data"]["num_workers"],
    )

    metrics_accum = SegmentationMetrics()
    n_vis_saved = 0
    n_vis_target = cfg["output"]["num_visualization_samples"]
    vis_dir = os.path.join(run_dir, "visualizations")
    pred_dir = os.path.join(run_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            logits = model(images.to(device))
            metrics_accum.update(logits, masks.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for i in range(images.shape[0]):
                if n_vis_saved >= n_vis_target:
                    break
                save_prediction_overlay(
                    images[i].numpy(), masks[i].numpy(), preds[i],
                    os.path.join(vis_dir, f"test_pred_{batch_idx}_{i}.png"),
                    ignore_index=IGNORE_INDEX,
                )
                n_vis_saved += 1

    results = metrics_accum.compute()
    cm = metrics_accum.get_confusion_matrix()

    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    with open(os.path.join(logs_dir, "test_metrics.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in results.items():
            if not isinstance(v, dict):
                f.write(f"{k},{v}\n")

    with open(os.path.join(logs_dir, "per_class_metrics.csv"), "w") as f:
        f.write("class,iou,f1,dice,precision,recall\n")
        for cname in CLASS_NAMES:
            f.write(
                f"{cname},{results['per_class_iou'][cname]:.6f},{results['per_class_f1'][cname]:.6f},"
                f"{results['per_class_dice'][cname]:.6f},{results['per_class_precision'][cname]:.6f},"
                f"{results['per_class_recall'][cname]:.6f}\n"
            )

    save_confusion_matrix_csv(cm, os.path.join(logs_dir, "confusion_matrix.csv"))
    save_confusion_matrix_figure(cm, CLASS_NAMES, os.path.join(vis_dir, "confusion_matrix.png"))

    logger.info(
        f"Test  mIoU={results['mean_iou']:.4f}  F1={results['macro_f1']:.4f}  "
        f"Dice={results['mean_dice']:.4f}  acc={results['pixel_accuracy']:.4f}"
    )
    logger.info(
        f"Marine Debris  IoU={results['marine_debris_iou']:.4f}  "
        f"F1={results['marine_debris_f1']:.4f}  "
        f"P={results['marine_debris_precision']:.4f}  "
        f"R={results['marine_debris_recall']:.4f}"
    )
    return results


def print_and_save_summary(cfg, experiment, encoder_desc, ssl_used, best_val_miou, test_results, run_dir, logger):
    lines = [
        f"Experiment:          {experiment}",
        f"Encoder:             {encoder_desc}",
        f"SSL:                 {'MAE-style masked reconstruction' if ssl_used else 'None'}",
        f"Decoder:             U-Net++",
        f"Classes:             {NUM_CLASSES}",
        "",
        f"Best val mIoU:       {best_val_miou:.4f}",
        f"Test mIoU:           {test_results['mean_iou']:.4f}",
        f"Test macro F1:       {test_results['macro_f1']:.4f}",
        f"Test mean Dice:      {test_results['mean_dice']:.4f}",
        "",
        "Marine Debris:",
        f"  IoU:               {test_results['marine_debris_iou']:.4f}",
        f"  F1:                {test_results['marine_debris_f1']:.4f}",
        f"  Precision:         {test_results['marine_debris_precision']:.4f}",
        f"  Recall:            {test_results['marine_debris_recall']:.4f}",
    ]
    logger.info("\nFINAL SUMMARY\n" + "\n".join(lines))

    summary_csv = os.path.join(run_dir, "summary.csv")
    with open(summary_csv, "w") as f:
        f.write("field,value\n")
        f.write(f"experiment,{experiment}\n")
        f.write(f"encoder,{encoder_desc}\n")
        f.write(f"ssl_method,{'MAE-style masked reconstruction' if ssl_used else 'None'}\n")
        f.write(f"decoder,U-Net++\n")
        f.write(f"num_classes,{NUM_CLASSES}\n")
        f.write(f"best_val_miou,{best_val_miou:.6f}\n")
        f.write(f"test_miou,{test_results['mean_iou']:.6f}\n")
        f.write(f"test_macro_f1,{test_results['macro_f1']:.6f}\n")
        f.write(f"test_mean_dice,{test_results['mean_dice']:.6f}\n")
        f.write(f"marine_debris_iou,{test_results['marine_debris_iou']:.6f}\n")
        f.write(f"marine_debris_f1,{test_results['marine_debris_f1']:.6f}\n")
        f.write(f"marine_debris_precision,{test_results['marine_debris_precision']:.6f}\n")
        f.write(f"marine_debris_recall,{test_results['marine_debris_recall']:.6f}\n")
    logger.info(f"Summary saved to {summary_csv}")


def main():
    args = parse_args()
    cfg = load_config(args.Config)
    set_seed(cfg["seed"], deterministic_cudnn=cfg.get("deterministic_cudnn", True))
    device = get_device()

    run_dir = setup_run_dir(cfg, args.experiment, args.Config)
    logger = build_console_logger("main", os.path.join(run_dir, "logs", "main_console.log"))
    logger.info(f"Device: {device}  Mode: {args.mode}  Experiment: {args.experiment}")
    log_gpu_environment(logger)

    if args.mode == "sanity":
        run_sanity(cfg, device, logger)
        return

    ssl_needed = experiment_needs_ssl(args.experiment)
    encoder = build_experiment_encoder(cfg, args.experiment, logger)
    encoder_desc = {
        "baseline": "ViT-S/16 (random init)",
        "pretrained": "ViT-S/16 (pretrained checkpoint)",
        "ssl_adapted": "ViT-S/16 (pretrained) + SSL-adapted",
        "ssl_from_scratch": "ViT-S/16 (random) + SSL from scratch",
    }[args.experiment]

    ssl_ckpt_path = None

    if args.mode in ("ssl", "full") and ssl_needed:
        ssl_ckpt_path = run_ssl_stage(cfg, args.experiment, encoder, device, run_dir, args.resume, logger)
    elif args.mode == "ssl" and not ssl_needed:
        logger.warning(f"Experiment '{args.experiment}' has no SSL stage.")
        return
    elif args.mode == "segmentation" and ssl_needed:
        existing = os.path.join(run_dir, "checkpoints", "ssl_best.pth")
        if not os.path.exists(existing):
            raise FileNotFoundError(f"SSL checkpoint not found at '{existing}'. Run --mode ssl first.")
        ssl_ckpt_path = existing
        logger.info(f"Reusing SSL checkpoint: {ssl_ckpt_path}")

    if ssl_ckpt_path is not None:
        ssl_transfer_model = MaskedAutoencoderSSL(
            encoder,
            decoder_embed_dim=cfg["ssl"]["decoder_embed_dim"],
            decoder_depth=cfg["ssl"]["decoder_depth"],
            decoder_num_heads=cfg["ssl"]["decoder_num_heads"],
            mask_ratio=cfg["ssl"]["mask_ratio"],
            norm_pix_loss=cfg["ssl"]["norm_pix_loss"],
        ).to(device)
        load_model_checkpoint_compatible(ssl_ckpt_path, ssl_transfer_model, device, logger)

    dataset_info = {"ssl_root": cfg["data"]["ssl_root"], "marida_root": cfg["data"]["marida_root"]}
    save_run_metadata(run_dir, cfg, args.experiment, device, dataset_info)

    best_val_miou = None
    if args.mode in ("segmentation", "full"):
        best_seg_ckpt = run_segmentation_stage(cfg, args.experiment, encoder, device, run_dir, args.resume, logger)
        best_val_miou = torch.load(best_seg_ckpt, map_location="cpu").get("best_metric", None)

    if args.mode in ("test", "full"):
        test_results = run_test_stage(cfg, args.experiment, device, run_dir, logger)
        if best_val_miou is None:
            best_ckpt = os.path.join(run_dir, "checkpoints", "segmentation_best.pth")
            best_val_miou = torch.load(best_ckpt, map_location="cpu").get("best_metric", 0.0) if os.path.exists(best_ckpt) else 0.0
        print_and_save_summary(cfg, args.experiment, encoder_desc, ssl_needed, best_val_miou, test_results, run_dir, logger)


if __name__ == "__main__":
    main()