import os
import glob
import pickle
from pathlib import Path

import hydra
from omegaconf import DictConfig

from dataset import load_audio, load_image
from features import process_audio, process_image


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
    eval_dir = cfg.paths.eval_dir
    models_dir = cfg.paths.models_dir

    print("Loading pre-trained models...")
    with open(os.path.join(models_dir, "audio_gmm.pkl"), "rb") as f:
        model_audio = pickle.load(f)
    with open(os.path.join(models_dir, "image_mlp.pkl"), "rb") as f:
        model_image = pickle.load(f)
    with open(os.path.join(models_dir, "fusion.pkl"), "rb") as f:
        model_fusion = pickle.load(f)

    # Gather all unique base names in the evaluation directory
    all_files = glob.glob(os.path.join(eval_dir, "*.*"))
    base_names = sorted(list(set(Path(f).stem for f in all_files)))

    if not base_names:
        print(f"No files found in {eval_dir}. Please check the directory.")
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
            fs, sig = load_audio(wav_path)
            feat_audio = process_audio(sig, fs, augment=False)  # NEVER augment eval data

            score_a = model_audio.predict_score([feat_audio])[0]
            decision_a = int(score_a > 0)
            results_audio.append((name, score_a, decision_a))

        # Process Image
        if has_image:
            img = load_image(png_path)
            feat_image = process_image(img, augment=False)  # NEVER augment eval data

            score_i = model_image.predict_score([feat_image])[0]
            decision_i = int(score_i > 0)
            results_image.append((name, score_i, decision_i))

        # Process Fusion (requires both modalities)
        if has_audio and has_image:
            score_f = model_fusion.predict_score([(feat_audio, feat_image)])[0]
            decision_f = int(score_f > 0)
            results_fusion.append((name, score_f, decision_f))

    # Output the files matching the requested naming conventions
    print("\nGenerating submission files...")
    if results_audio:
        write_results("audio_GMM.txt", results_audio)
    if results_image:
        write_results("image_MLP.txt", results_image)
    if results_fusion:
        write_results("fusion_Late.txt", results_fusion)


if __name__ == "__main__":
    main()
