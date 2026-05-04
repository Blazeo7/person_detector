import hydra
import numpy as np
import os
import pickle
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from dataset.data_loader import DataLoader
from sklearn.metrics import (
    DetCurveDisplay,
    det_curve,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from utils.logger import Logger
from validation import session_kfold_splits

import matplotlib.pyplot as plt
import seaborn as sns



def train_modality(cfg: DictConfig, fold, fold_idx: int, modality: str):
    """Instantiate, fit, and return a model for one modality on one fold."""
    model = instantiate(cfg.models[modality])

    if modality == "image":
        X_train, y_train = fold.train_images, fold.train_labels
        X_val, y_val = fold.val_images, fold.val_labels
    else:
        X_train, y_train = fold.train_audios, fold.train_labels
        X_val, y_val = fold.val_audios, fold.val_labels

    model.fit(X_train, y_train)

    # ── Probability distribution diagnostic ──────────────────────────────
    train_probs_full = model.predict_proba(X_train)
    train_probs = train_probs_full[:, 1] if train_probs_full.ndim > 1 else train_probs_full

    val_probs_full = model.predict_proba(X_val)
    val_probs = val_probs_full[:, 1] if val_probs_full.ndim > 1 else val_probs_full

    print(f"  [{modality}] Probability stats (TRAIN)")
    print(
        f"    target     n={( y_train==1).sum()}  "
        f"mean={train_probs[y_train==1].mean():.3f}  "
        f"std={train_probs[y_train==1].std():.3f}  "
        f"min={train_probs[y_train==1].min():.3f}  "
        f"max={train_probs[y_train==1].max():.3f}"
    )
    print(
        f"    non-target n={(y_train==0).sum()}  "
        f"mean={train_probs[y_train==0].mean():.3f}  "
        f"std={train_probs[y_train==0].std():.3f}  "
        f"min={train_probs[y_train==0].min():.3f}  "
        f"max={train_probs[y_train==0].max():.3f}"
    )

    print(f"  [{modality}] Probability stats (VAL)")
    print(
        f"    target     n={(y_val==1).sum()}  "
        f"mean={val_probs[y_val==1].mean():.3f}  "
        f"std={val_probs[y_val==1].std():.3f}  "
        f"min={val_probs[y_val==1].min():.3f}  "
        f"max={val_probs[y_val==1].max():.3f}"
    )
    print(
        f"    non-target n={(y_val==0).sum()}  "
        f"mean={val_probs[y_val==0].mean():.3f}  "
        f"std={val_probs[y_val==0].std():.3f}  "
        f"min={val_probs[y_val==0].min():.3f}  "
        f"max={val_probs[y_val==0].max():.3f}"
    )

    def _metrics(X, y, split: str) -> dict:
        probs_full = model.predict_proba(X)
        probs = probs_full[:, 1] if probs_full.ndim > 1 else probs_full
        preds = (probs >= model.threshold).astype(int)  # Respects dynamic threshold

        try:
            auc_val = roc_auc_score(y, probs)
        except ValueError:
            auc_val = 0.0

        f1 = f1_score(y, preds, zero_division=0)
        cm = confusion_matrix(y, preds, labels=[0, 1])

        return {
            f"{split}_auc": auc_val,
            f"{split}_f1": f1,
            f"{split}_cm": cm,
            f"{split}_y_true": y,
            f"{split}_probs": probs,
        }

    metrics = {**_metrics(X_train, y_train, "train"), **_metrics(X_val, y_val, "val")}
    return model, metrics


def train_fusion(cfg, fold, fold_idx, model_audio, model_image):
    # Dynamically instantiate from config
    fusion_model = instantiate(cfg.models.fusion, model_audio=model_audio, model_image=model_image)

    X_train_audio = fold.train_audios
    X_train_image = fold.train_images
    y_train = fold.train_labels

    X_val_audio = fold.val_audios
    X_val_image = fold.val_images
    y_val = fold.val_labels

    # Zip the features
    X_train_fusion = list(zip(X_train_audio, X_train_image))
    X_val_fusion = list(zip(X_val_audio, X_val_image))

    fusion_model.fit(X_train_fusion, y_train)

    # Evaluate (Train)
    train_probs_full = fusion_model.predict_proba(X_train_fusion)
    train_probs_1d = train_probs_full[:, 1] if train_probs_full.ndim > 1 else train_probs_full
    train_preds = fusion_model.predict(X_train_fusion)

    try:
        train_auc = roc_auc_score(y_train, train_probs_1d)
    except ValueError:
        train_auc = 0.0

    train_f1 = f1_score(y_train, train_preds, zero_division=0)
    train_cm = confusion_matrix(y_train, train_preds, labels=[0, 1])

    # Evaluate (Validation)
    val_probs_full = fusion_model.predict_proba(X_val_fusion)
    val_probs_1d = val_probs_full[:, 1] if val_probs_full.ndim > 1 else val_probs_full
    val_preds = fusion_model.predict(X_val_fusion)

    try:
        val_auc = roc_auc_score(y_val, val_probs_1d)
    except ValueError:
        val_auc = 0.0

    val_f1 = f1_score(y_val, val_preds, zero_division=0)
    val_cm = confusion_matrix(y_val, val_preds, labels=[0, 1])

    metrics = {
        "train_auc": train_auc,
        "train_f1": train_f1,
        "train_cm": train_cm,
        "val_auc": val_auc,
        "val_f1": val_f1,
        "val_cm": val_cm,
        "val_y_true": y_val,
        "val_probs": val_probs_1d,
    }
    return fusion_model, metrics


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    loader = DataLoader()
    loader.load_data(
        target_dirs=list(cfg.paths.target_train),
        non_target_dirs=list(cfg.paths.non_target_train),
    )
    print(
        f"Loaded {len(loader.samples)} samples "
        f"({sum(s.label for s in loader.samples)} target, "
        f"{sum(1 - s.label for s in loader.samples)} non-target)"
    )

    # ------------------------------------------------------------------ #
    # Cross-validation
    # ------------------------------------------------------------------ #
    fold_metrics = {"image": [], "audio": [], "fusion": []}
    best_models = {"image": None, "audio": None, "fusion": None}
    best_val_acc = {"image": 0.0, "audio": 0.0, "fusion": 0.0}

    saved_models_dir = cfg.paths.models_dir
    os.makedirs(saved_models_dir, exist_ok=True)

    for fold_i, fold in enumerate(
        session_kfold_splits(
            loader,
            n_splits=cfg.training.cv_splits,
            n_augmentations=cfg.training.n_augmentations,
            feature_cfg=cfg.features.image,
        )
    ):
        print(f"Fold {fold_i}")

        fold_trained_models = {}
        for modality in ("image", "audio"):
            model, metrics = train_modality(cfg, fold, fold_i, modality)
            fold_trained_models[modality] = model
            fold_metrics[modality].append(metrics)
            Logger(True).warning(
                f"  [{modality}] "
                f"train_auc={metrics['train_auc']:.4f}  train_f1={metrics['train_f1']:.4f} | "
                f"val_auc={metrics['val_auc']:.4f}    val_f1={metrics['val_f1']:.4f}"
            )

            if metrics["val_auc"] > best_val_acc[modality]:
                best_val_acc[modality] = metrics["val_auc"]
                best_models[modality] = model

        model_fusion, fusion_metrics = train_fusion(
            cfg, fold, fold_i, fold_trained_models["audio"], fold_trained_models["image"]
        )
        fold_metrics["fusion"].append(fusion_metrics)
        print(
            f"  [fusion] "
            f"train_auc={fusion_metrics['train_auc']:.4f}  train_f1={fusion_metrics['train_f1']:.4f} | "
            f"val_auc={fusion_metrics['val_auc']:.4f}    val_f1={fusion_metrics['val_f1']:.4f}"
        )

        if fusion_metrics["val_auc"] > best_val_acc["fusion"]:
            best_val_acc["fusion"] = fusion_metrics["val_auc"]
            best_models["fusion"] = model_fusion
        # ---------------------------------

    print("\n══ CV Summary ══")

    # Setup Figures ---
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(16, 5))
    fig_cm.suptitle("Average Validation Confusion Matrices (Row-Normalized %)", fontsize=16)

    # Setup for DET Curve ---
    fig_det, ax_det = plt.subplots(figsize=(8, 8))
    colors = {"image": "blue", "audio": "orange", "fusion": "green"}

    for i, modality in enumerate(("image", "audio", "fusion")):
        print(f"\n  [{modality.upper()}]")

        # Metrics Printout
        for metric in ("auc", "f1"):
            vals = [m[f"val_{metric}"] for m in fold_metrics[modality]]
            print(
                f"    val_{metric}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  best={np.max(vals):.4f}"
            )

        # Normalized Confusion Matrix
        val_cms = [m["val_cm"] for m in fold_metrics[modality]]
        avg_cm = np.mean(val_cms, axis=0)

        # Normalize by rows (true labels) to get percentages
        avg_cm_norm = avg_cm.astype("float") / avg_cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            avg_cm_norm,
            annot=True,
            fmt=".1%",
            cmap="Blues",
            ax=axes_cm[i],
            cbar=False,
            annot_kws={"size": 14},
            vmin=0.0,
            vmax=1.0,
        )
        axes_cm[i].set_title(f"{modality.capitalize()} Model")
        axes_cm[i].set_xlabel("Predicted Label")
        axes_cm[i].set_ylabel("True Label")

        # Pooled DET Curve
        # Concatenate true labels and probs across all folds to create one smooth, high-res curve
        all_y_true = np.concatenate([m["val_y_true"] for m in fold_metrics[modality]])
        all_probs = np.concatenate([m["val_probs"] for m in fold_metrics[modality]])

        # Calculate FPR and FNR
        fpr, fnr, _ = det_curve(all_y_true, all_probs)

        # Use sklearn's built-in display to automatically apply the normal-deviate axis scaling
        display = DetCurveDisplay(fpr=fpr, fnr=fnr, estimator_name=modality.capitalize())
        display.plot(ax=ax_det, color=colors[modality], linewidth=2.5)

    # Finalize CM Plot
    fig_cm.tight_layout()

    # Finalize DET Plot
    ax_det.set_title(
        "Detection Error Tradeoff (DET) Curve\n(Pooled across Validation Folds)", fontsize=14
    )
    # Add a fine grid to make reading the logarithmic-style scale easier
    ax_det.grid(alpha=0.4, which="both", linestyle="--")
    ax_det.legend(loc="upper right", fontsize=12)

    plt.show()

    # ------------------------------------------------------------------ #
    # Final Model Training (100% of Data)
    # ------------------------------------------------------------------ #
    print("\n══ Training Final Models on 100% of Data ══")

    def get_optimal_threshold(y_true, y_probs):
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = np.divide(
            2 * (precisions * recalls),
            (precisions + recalls),
            out=np.zeros_like(precisions),
            where=(precisions + recalls) != 0,
        )
        best_idx = np.argmax(f1_scores)
        return pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else 0.5

    X_all_images = np.concatenate((fold.train_images, fold.val_images), axis=0)
    X_all_audios = np.concatenate((fold.train_audios, fold.val_audios), axis=0)
    y_all = np.concatenate((fold.train_labels, fold.val_labels), axis=0)

    final_models = {}

    # Train Final Base Models
    for modality in ["audio", "image"]:
        print(f"  Training Final {modality.capitalize()} Model...")
        model = instantiate(cfg.models[modality])
        X_all = X_all_audios if modality == "audio" else X_all_images

        model.fit(X_all, y_all)

        probs_full = model.predict_proba(X_all)
        probs_1d = probs_full[:, 1] if probs_full.ndim > 1 else probs_full

        best_thr = get_optimal_threshold(y_all, probs_1d)
        model.threshold = float(best_thr)
        print(f"    -> Best Threshold Found: {best_thr:.4f}")

        final_models[modality] = model

    # Train Final Fusion Model
    print("  Training Final Fusion Model...")
    fusion_model = instantiate(
        cfg.models.fusion, model_audio=final_models["audio"], model_image=final_models["image"]
    )
    X_all_fusion = list(zip(X_all_audios, X_all_images))

    fusion_model.fit(X_all_fusion, y_all)

    fusion_probs_full = fusion_model.predict_proba(X_all_fusion)
    fusion_probs_1d = fusion_probs_full[:, 1] if fusion_probs_full.ndim > 1 else fusion_probs_full

    best_fusion_thr = get_optimal_threshold(y_all, fusion_probs_1d)
    fusion_model.threshold = float(best_fusion_thr)
    print(f"    -> Best Fusion Threshold Found: {best_fusion_thr:.4f}")

    final_models["fusion"] = fusion_model

    # Save Final Models
    for modality, model in final_models.items():
        path = os.path.join(saved_models_dir, f"hope_{modality}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved final deployed {modality} model → {path}")


if __name__ == "__main__":
    main()
