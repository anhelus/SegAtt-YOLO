"""Batch training over multiple YOLO model variants.

Reads configuration from train_config.yaml and runs training
for each model variant.

Typical usage::

    python scripts/batch_train.py --config train_config.yaml
    python scripts/batch_train.py --config train_config.yaml --model yolo11n-ema
    python scripts/batch_train.py --config train_config.yaml --model yolo11 --scale s
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from segatt_yolo import YOLO


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _model_yaml_path(model_name: str) -> Path:
    """Resolve model YAML path (supports both model name and explicit paths)."""
    repo_root = Path(__file__).resolve().parent.parent
    if model_name.endswith((".yaml", ".yml")):
        p = Path(model_name)
        return p if p.is_absolute() else repo_root / p
    return repo_root / "segatt_yolo" / "cfg" / "models" / "11" / f"{model_name}.yaml"


def _yaml_with_scale(yaml_path: Path, scale: str) -> str:
    """
    Load YAML, inject scale, write to temp file with scale in filename, return temp file path.
    Caller must clean up the temp file.
    Filename must match guess_model_scale() regex: yolo<ver><scale>_*
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["scale"] = scale
    # Filename must be like yolo11s_<random>.yaml for guess_model_scale() to extract 's'
    stem = yaml_path.stem  # e.g., "yolo11"
    tmp = tempfile.NamedTemporaryFile(mode="w", prefix=f"{stem}{scale}_", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    return tmp.name


def _target_folder_name(run_name: str, scale: str) -> str:
    """
    Convert training folder name to XAI/Infer/Viz config folder name.
    
    Training names (from train_config.yaml):
    - yolo11 -> yolo11n
    - yolo11_gam_min -> yolo11n-gam_min
    - yolo11_gam_full -> yolo11n-gam_full
    - yolo11_simam -> yolo11n-simam
    - yolo11_gam_bbone_p3p4 -> yolo11n-gam_bbone_p3p4
    - yolo26 -> yolo26n
    
    Pattern: insert scale after base model (yolo11/yolo26), 
    first underscore becomes hyphen, rest stay underscore.
    """
    # Handle yolo11* variants
    if run_name.startswith("yolo11"):
        rest = run_name[6:]  # after "yolo11"
        if not rest:
            return f"yolo11{scale}"
        # First underscore -> hyphen, rest stay underscore
        parts = rest.split('_', 1)
        if len(parts) == 1:
            return f"yolo11{scale}{rest}"
        return f"yolo11{scale}-{parts[1]}"
    
    # Handle yolo26
    if run_name.startswith("yolo26"):
        rest = run_name[6:]
        if not rest:
            return f"yolo26{scale}"
        parts = rest.split('_', 1)
        if len(parts) == 1:
            return f"yolo26{scale}{rest}"
        return f"yolo26{scale}-{parts[1]}"
    
    # Fallback: just append scale
    return f"{run_name}{scale}"


def validate_config(config: Dict) -> None:
    """Validate configuration structure."""
    if "models" not in config or not config["models"]:
        raise ValueError("Config must contain at least one model under 'models'.")
    if "datasets" not in config or not config["datasets"]:
        raise ValueError("Config must contain 'datasets' list.")
    if "training" not in config:
        raise ValueError("Config must contain 'training'.")


def batch_run(
    config: Dict,
    model_filter: Optional[str] = None,
    dataset_filter: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
    data_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_override: Optional[int] = None,
    imgsz_override: Optional[int] = None,
    device_override: Optional[str] = None,
    name_override: Optional[str] = None,
    scale_override: Optional[str] = None,
    auto_rename: bool = False,
) -> None:
    """Run training over all specified model variants and datasets."""
    models = config["models"]
    datasets = config["datasets"]
    training_cfg = config["training"]

    epochs = epochs_override if epochs_override is not None else training_cfg.get("epochs", 100)
    batch = batch_override if batch_override is not None else training_cfg.get("batch", 16)
    imgsz = imgsz_override if imgsz_override is not None else training_cfg.get("imgsz", 640)
    fraction = training_cfg.get("fraction", 1.0)
    device = device_override if device_override is not None else training_cfg.get("device", None)
    patience = training_cfg.get("patience", 50)
    scale = scale_override if scale_override is not None else training_cfg.get("scale", "n")

    from tqdm import tqdm as tqdm_bar

    filtered_datasets = [
        d for d in datasets
        if not dataset_filter or d.get("name") == dataset_filter
    ]
    if not filtered_datasets:
        print("No datasets match the filter.")
        return

    for ds in filtered_datasets:
        ds_name = ds["name"]
        # Apply data_override only if dataset_filter is set (single dataset mode)
        # Otherwise use each dataset's own path
        if data_override is not None and dataset_filter is not None:
            ds_path = data_override
        else:
            ds_path = ds["path"]
        project = ds.get("project", f"runs/{ds_name}")
        abs_project = str(Path(project).resolve())
        os.makedirs(abs_project, exist_ok=True)

        def _model_name(m):
            return m if isinstance(m, str) else m.get("name", m.get("model", str(m)))
        total = sum(
            1 for m in models
            if not model_filter or _model_name(m) == model_filter
        )
        if total == 0:
            continue

        pbar = tqdm_bar(total=total, desc=f"Train {ds_name}", disable=verbose)
        trained = 0

        for entry in models:
            model_name = _model_name(entry)

            if model_filter and model_name != model_filter:
                continue

            trained += 1

            sep = "=" * 70
            if verbose:
                print(f"\n{sep}", flush=True)
                print(f"  [{trained}/{total}] Training: dataset={ds_name}  model={model_name}  scale={scale}", flush=True)
                print(f"{sep}\n", flush=True)

            if dry_run:
                if verbose:
                    print(f"  [DRY RUN] (would train)", flush=True)
                pbar.update(1)
                continue

            try:
                # Resolve model YAML and inject scale via temp file
                yaml_path = _model_yaml_path(model_name)
                scaled_yaml = _yaml_with_scale(yaml_path, scale)
                try:
                    model = YOLO(scaled_yaml)
                    run_name = name_override if name_override is not None else model_name
                    model.train(
                        data=ds_path,
                        project=abs_project,
                        name=run_name,
                        epochs=epochs,
                        batch=batch,
                        imgsz=imgsz,
                        fraction=fraction,
                        device=device,
                        patience=patience,
                        exist_ok=True,
                    )
                    if verbose:
                        print(f"  ok {ds_name}/{model_name} (scale={scale})", flush=True)
                    
                    # Auto-rename folder for XAI/Infer/Viz compatibility
                    if auto_rename:
                        src = Path(abs_project) / run_name
                        target_name = _target_folder_name(run_name, scale)
                        dst = Path(abs_project) / target_name
                        if src.exists() and not dst.exists():
                            import shutil
                            shutil.copytree(src, dst)
                            if verbose:
                                print(f"  📁 Auto-renamed: {src.name} -> {dst.name}", flush=True)
                finally:
                    os.unlink(scaled_yaml)
            except Exception as e:
                if verbose:
                    print(f"  fail {ds_name}/{model_name}: {e}", flush=True)

            pbar.update(1)

        pbar.close()
        print(f"Done: {trained} model(s) trained for {ds_name} -> {project}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch training over multiple YOLO model variants."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to training configuration YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Only train this model (name). E.g. 'yolo11n'.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Only train on this dataset (name). E.g. 'tomatoes'.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Custom data YAML path (overrides dataset path from config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config training.epochs).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size (overrides config training.batch).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size (overrides config training.imgsz).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (overrides config training.device).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom run name (subdirectory under project). Defaults to model name.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=["n", "s", "m", "l", "x"],
        help="Model scale (n/s/m/l/x). Overrides config training.scale. Default: n",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output instead of progress bar",
    )
    parser.add_argument(
        "--auto-rename",
        action="store_true",
        help="Automatically copy trained model folder with scale suffix for XAI/Infer/Viz compatibility (e.g., yolo11_gam_min -> yolo11n-gam_min)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    batch_run(
        config,
        model_filter=args.model,
        dataset_filter=args.dataset,
        dry_run=args.dry_run,
        verbose=args.verbose,
        data_override=args.data,
        epochs_override=args.epochs,
        batch_override=args.batch,
        imgsz_override=args.imgsz,
        device_override=args.device,
        name_override=args.name,
        scale_override=args.scale,
        auto_rename=args.auto_rename,
    )


if __name__ == "__main__":
    main()
