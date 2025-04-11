#!/usr/bin/env python3
# this will plot ALL the models in the validation directory
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime


def plot_validation_results(validation_dir):
    # Dictionary to store results for each model
    model_results = {}

    # Iterate through model folders in the validation directory
    for model_folder in os.listdir(validation_dir):
        model_path = os.path.join(validation_dir, model_folder)
        if not os.path.isdir(model_path):
            print(f"'{model_path}' is not found")
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
            # matches the literal text checkpoint- and then (\d+) matches one or more digits
            checkpoint_match = re.search(r"checkpoint-(\d+)", line)
            # matches one or more digits, then the literal . , then one or more digits
            accuracy_match = re.search(r"Final Accuracy: (\d+\.\d+)", line)

            if checkpoint_match and accuracy_match:
                # return the contents of the first capturing group
                checkpoint = int(checkpoint_match.group(1))
                accuracy = float(accuracy_match.group(1))
                results.append((checkpoint, accuracy))

        if results:
            model_results[model_folder] = sorted(results)

    # Plot results
    plt.figure(figsize=(10, 6))

    for model_name, results in model_results.items():
        checkpoints, accuracies = zip(*results)
        plt.plot(checkpoints, accuracies, marker="o", label=model_name)

    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    plt.xlabel("Checkpoint")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy by Checkpoint")

    plt.grid(True)
    plt.legend()
    # add current time to filename
    plt.savefig(f"validation_results - {current_time}.png")


if __name__ == "__main__":
    # this current directory
    plot_validation_results("./")
