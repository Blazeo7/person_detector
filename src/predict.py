import os
import glob
import pickle
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

# Unified DataLoader and Processors
from dataset.data_loader import DataLoader
from features.audio_features import AudioProcessor
from features.image_features import ImageProcessor


def write_results(filename, results):
    """
    Writes results to ASCII file.
    Format: segment_name score hard_decision
    """
    with open(filename, "w") as f:
        for name, score, decision in results:
            f.write(f"{name} {score:.6f} {decision}\n")
    print(f"Results saved to {filename}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    eval_dir = os.path.join(orig_cwd, cfg.paths.eval_dir)
    models_dir = os.path.join(orig_cwd, cfg.paths.models_dir)

    print("Loading pre-trained models...")
    try:
        with open(os.path.join(models_dir, "final_audio.pkl"), "rb") as f:
            model_audio = pickle.load(f)
        with open(os.path.join(models_dir, "final_image.pkl"), "rb") as f:
            model_image = pickle.load(f)
        with open(os.path.join(models_dir, "final_fusion.pkl"), "rb") as f:
            model_fusion = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find model files in {models_dir}. Did you run train.py first?")
        return

    # Initialize Processors (Loaders are handled via DataLoader static methods)
    audio_processor = AudioProcessor(fs=16000)
    image_processor = ImageProcessor()

    # Gather all unique base names in the evaluation directory
    all_files = glob.glob(os.path.join(eval_dir, "*.*"))
    base_names = sorted(list(set(Path(f).stem for f in all_files)))

    if not base_names:
        print(f"No files found in {eval_dir}. Please check the directory path in your config.")
        return

    print(f"Found {len(base_names)} segments to evaluate. Processing...")

    results_audio = []
    results_image = []
    results_fusion = []

    for name in base_names:
        wav_path = os.path.join(eval_dir, f"{name}.wav")
        png_path = os.path.join(eval_dir, f"{name}.png")

        has_audio = os.path.exists(wav_path)
        has_image = os.path.exists(png_path)

        feat_audio = None
        feat_image = None

        # Process Audio
        if has_audio:
            # Using the static method directly from the unified DataLoader
            fs, sig = DataLoader.load_single_audio(wav_path, target_sr=16000)

            # NEVER augment eval data
            feat_audio = audio_processor.process(sig, augment=False)

            # Predict using predict_proba, slicing [0, 1] for the target class probability
            score_a = model_audio.predict_proba([feat_audio])[0, 1]
            decision_a = int(score_a > 0.5)
            results_audio.append((name, score_a, decision_a))

        # Process Image
        if has_image:
            # Using the static method directly from the unified DataLoader
            img = DataLoader.load_single_image(png_path)

            # NEVER augment eval data
            feat_image = image_processor.process(img, augment=False, feature_cfg=cfg.features.image)

            # Predict using model_image!
            score_i = model_image.predict_proba([feat_image])[0, 1]
            decision_i = int(score_i > 0.5)
            results_image.append((name, score_i, decision_i))

        # Process Fusion
        if has_audio and has_image:
            # Handle both 1D and 2D arrays safely
            prob_f_array = model_fusion.predict_proba([(feat_audio, feat_image)])
            score_f = prob_f_array[0, 1] if prob_f_array.ndim > 1 else prob_f_array[0]

            # Uniform decision boundary
            decision_f = int(score_f > 0.5)
            results_fusion.append((name, score_f, decision_f))

    # Output files
    print("\nGenerating submission files...")
    if results_audio:
        write_results("audio_SVM.txt", results_audio)
    if results_image:
        write_results("image_SVM.txt", results_image)
    if results_fusion:
        write_results("fusion_Late_LR.txt", results_fusion)


if __name__ == "__main__":
    main()
