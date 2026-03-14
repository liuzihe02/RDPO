#!/usr/bin/env python3
# this will plot ALL the models in the validation directory
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

MODEL_RUNS = [
    "train-qwen2.5-0.5b-genrm-dpo-eval-gsm8k,math-500",
    "train-qwen2.5-0.5b-genrm-rdpo-eval-gsm8k,math-500",
    "train-qwen2.5-0.5b-genrm-sft-veri-eval-gsm8k,math-500",
    "train-qwen2.5-0.5b-genrm-sft-no_veri-eval-gsm8k,math-500",
]

DISPLAY_NAMES = {
    "train-qwen2.5-0.5b-genrm-dpo-eval-gsm8k,math-500": "DPO",
    "train-qwen2.5-0.5b-genrm-rdpo-eval-gsm8k,math-500": r"CDPO ($\alpha$=0.5)",
    "train-qwen2.5-0.5b-genrm-sft-veri-eval-gsm8k,math-500": "SFT (with verification)",
    "train-qwen2.5-0.5b-genrm-sft-no_veri-eval-gsm8k,math-500": "SFT (no verification)",
}


def plot_evaluate_results(validation_dir):
    # Dictionary to store results for each model and dataset
    # =================
    # THIS NEED TO BE CONFIGURED BY YOU!!
    # =================
    dataset_results = {"gsm8k": {}, "math-500": {}}
    # for each dataset, we search for it
    for dataset_name in dataset_results.keys():
        # Iterate through model folders in the validation directory
        for model_folder in MODEL_RUNS:
            model_path = os.path.join(validation_dir, model_folder)

            print(f"modelpath is {model_path}")
            if not os.path.isdir(model_path):
                print(f"'{model_path}' is not a directory")
                continue

            # Path to the validation summary file
            summary_file = os.path.join(model_path, "validation_summary.txt")
            if not os.path.exists(summary_file):
                print(f"'{summary_file}' is not found")
                continue

            # Read the summary file
            with open(summary_file, "r") as f:
                content = f.read()

            # Extract checkpoint and accuracy data
            results = []
            for line in content.strip().split("\n"):
                # Extract checkpoint, dataset, and accuracy in one regex
                match = re.search(
                    r"/checkpoint-(\d+)/([a-zA-Z0-9-]+).*Final Accuracy: (\d+\.\d+)",
                    line,
                )

                if match:
                    checkpoint = int(match.group(1))
                    dataset_name = match.group(2)
                    accuracy = float(match.group(3))

                    print(f"zihe {checkpoint}, {dataset_name}, {accuracy}")

                    # Initialize model in dataset if needed
                    if model_folder not in dataset_results[dataset_name]:
                        dataset_results[dataset_name][model_folder] = []

                    # Add result
                    dataset_results[dataset_name][model_folder].append(
                        (checkpoint, accuracy)
                    )

    # Create a plot for each dataset
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    for dataset_name, models in dataset_results.items():
        plt.figure(figsize=(10, 6))

        for model_name, results in models.items():
            # Sort results by checkpoint
            results.sort()
            if results:
                checkpoints, accuracies = zip(*results)
                # Normalize to epochs (2 epochs total, save_steps=0.1 means every 0.2 epochs)
                max_ckpt = max(checkpoints)
                epochs = [c / max_ckpt * 2.0 for c in checkpoints]
                plt.plot(epochs, accuracies, marker="o", label=DISPLAY_NAMES.get(model_name, model_name))

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{dataset_name} Validation Accuracy by Checkpoint")
        plt.grid(True)
        plt.legend()
        # add current time to filename
        plt.savefig(
            f"{validation_dir}/{dataset_name}_validation_results_{current_time}.png"
        )
        plt.close()


if __name__ == "__main__":
    # this current directory
    plot_evaluate_results("../evaluate-results/")
