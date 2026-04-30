import os
import pickle
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
from collections import defaultdict

from dataset import Sample, load_samples, load_audio, load_image, Modality, Label
from features import process_audio, process_image
from models.base import BaseDetector
from validation import session_kfold_splits


def prepare_paired_dataset(samples: list[Sample]):
    paired = defaultdict(dict)
    for s in samples:
        paired[s.name][s.modality] = s
        paired[s.name]["label"] = 1 if s.label == Label.TARGET else 0
        paired[s.name]["sample_obj"] = s
    return [v for v in paired.values() if Modality.AUDIO in v and Modality.IMAGE in v]


def extract_features(paired_data, augment=False):
    print(f"Extracting features (Augmentation: {augment})...")
    X_audio, X_image, y, raw_samples = [], [], [], []
    for pair in paired_data:
        # Audio
        fs, sig = load_audio(pair[Modality.AUDIO].path)
        X_audio.append(process_audio(sig, fs, augment=augment))

        # Image
        img = load_image(pair[Modality.IMAGE].path)
        X_image.append(process_image(img, augment=augment))

        y.append(pair["label"])
        raw_samples.append(pair["sample_obj"])
    return X_audio, X_image, np.array(y), raw_samples


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()

    models_dir = os.path.join(orig_cwd, cfg.paths.models_dir)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load and prepare data
    print("Loading data paths...")
    target_dirs = [os.path.join(orig_cwd, d) for d in cfg.paths.target_train]
    non_target_dirs = [os.path.join(orig_cwd, d) for d in cfg.paths.non_target_train]

    audio_samples = load_samples(target_dirs, non_target_dirs, Modality.AUDIO)
    image_samples = load_samples(target_dirs, non_target_dirs, Modality.IMAGE)

    paired_data = prepare_paired_dataset(audio_samples + image_samples)

    # Extract features (apply augmentation ONLY if specified in config)
    X_audio, X_image, y, raw_samples = extract_features(paired_data, augment=cfg.training.augment)
    X_fusion = list(zip(X_audio, X_image))

    # Map config keys to actual data
    data_map = {"audio": X_audio, "image": X_image, "fusion": X_fusion}

    # Cross-Validation Loop
    if cfg.training.cv_splits > 1:
        print(f"\n--- Running {cfg.training.cv_splits}-Fold Session-Aware Cross Validation ---")

        for fold, (train_idx, val_idx) in enumerate(
            session_kfold_splits(raw_samples, n_splits=cfg.training.cv_splits)
        ):
            print(f"\nFold {fold + 1}")
            fold_models = {}

            # Train Base Models (Audio, Image)
            for mod_key in ["audio", "image"]:
                if mod_key in cfg.models:
                    model: BaseDetector = instantiate(cfg.models[mod_key])
                    X_train = [data_map[mod_key][i] for i in train_idx]
                    X_val = [data_map[mod_key][i] for i in val_idx]

                    model.fit(X_train, y[train_idx])
                    acc = np.mean(model.predict(X_val) == y[val_idx])
                    print(f"  {mod_key.capitalize()} Val Accuracy: {acc:.4f}")
                    fold_models[mod_key] = model

            # Train Fusion Model (Requires trained base models)
            if "fusion" in cfg.models and "audio" in fold_models and "image" in fold_models:
                # Hydra cleverly passes the already instantiated fold_models as kwargs
                fusion_model = instantiate(
                    cfg.models.fusion, model_audio=fold_models["audio"], model_image=fold_models["image"]
                )
                X_train_f = [data_map["fusion"][i] for i in train_idx]
                X_val_f = [data_map["fusion"][i] for i in val_idx]

                fusion_model.fit(X_train_f, y[train_idx])
                acc = np.mean(fusion_model.predict(X_val_f) == y[val_idx])
                print(f"  Fusion Val Accuracy: {acc:.4f}")

    # Final Training on ALL Data
    print("\n--- Training Final Models ---")
    final_models = {}

    for mod_key in ["audio", "image"]:
        if mod_key in cfg.models:
            print(f"Training {mod_key} model on all data...")
            model = instantiate(cfg.models[mod_key], verbose=True)
            model.fit(data_map[mod_key], y)
            final_models[mod_key] = model

            # Save base model
            save_path = os.path.join(models_dir, f"{mod_key}_model.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(model, f)

    if "fusion" in cfg.models and "audio" in final_models and "image" in final_models:
        print("Training fusion model on all data...")
        fusion_model = instantiate(
            cfg.models.fusion,
            model_audio=final_models["audio"],
            model_image=final_models["image"],
            verbose=True,
        )
        fusion_model.fit(data_map["fusion"], y)

        # Save fusion model
        save_path = os.path.join(models_dir, "fusion_model.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(fusion_model, f)

    print(f"\nTraining complete! Models saved to {models_dir}")


if __name__ == "__main__":
    main()
