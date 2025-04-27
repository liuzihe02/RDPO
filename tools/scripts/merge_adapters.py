"""
Merge a PEFT adapter into its base model for a single checkpoint directory.

Usage:
    python merge_adapter.py --checkpoint-dir /path/to/checkpoint_dir [--base-model MODEL_NAME]
"""

import argparse
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re

# Files to keep after merging
# regex patterns
# ^ matches the start of string, \. means a literal dot, and $ means end of string
KEEP_PATTERNS = [
    r"^added_tokens\.json$",
    r"^config\.json$",  # model config
    r"^generation_config\.json$",
    r"^merges\.txt$",
    r"^model.*\.safetensors$",  # model weights with any characters between
    r"^model\.safetensors\.index\.json$",  # model index files
    r"^special_tokens_map\.json$",
    r"^tokenizer_config\.json$",
    r"^tokenizer\.json$",  # tokenizer files
    r"^vocab\.json$",
]


def merge_adapters(checkpoint_dir):
    # 1) Load the base model from using the adapter config
    adapter_path = checkpoint_dir + "/adapter_config.json"

    if os.path.exists(adapter_path):
        # indeed do merging base+adapter for this checkpoint dir
        print("=" * 50)
        print(f"Doing merge_adapters for checkpoint: {checkpoint_dir}")
        print("=" * 50)

        # get the config from the adapter config
        with open(adapter_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Extract the base model name/path
        base_model = cfg.get("base_model_name_or_path")

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        # 2) Load the trained adapter from the checkpoint directory
        trained = PeftModel.from_pretrained(base, checkpoint_dir)
        # 3) Merge and unload the adapter weights
        merged = trained.merge_and_unload()
        # 4) Save the merged model and tokenizer back to the checkpoint directory
        merged.save_pretrained(checkpoint_dir)
        AutoTokenizer.from_pretrained(checkpoint_dir).save_pretrained(checkpoint_dir)
        print(f"Merged model saved to {checkpoint_dir}")

        # Prune files not matching KEEP_PATTERNS
        # THIS IS BECAUSE THIS RUNS INTO ERRORS FOR VLLM; WE ONLY KEEP THE ESSENTIAL STUFF
        for fname in os.listdir(checkpoint_dir):
            fpath = os.path.join(checkpoint_dir, fname)
            # Check if file matches any regex pattern
            # this means we delete the original adapter weights, as keeping this causes problems
            keep = any(re.match(pat, fname) for pat in KEEP_PATTERNS)
            if not keep:
                # remove this extra file
                os.remove(fpath)

        print(
            f"Pruned directory, kept only merged model and tokenizer: {checkpoint_dir}"
        )

    # this is likely the initial checkpoint
    # no need to merge adapters, it is already in the right safetensors
    else:
        print("=" * 50)
        print(f"skipping merge_adapters for checkpoint: {checkpoint_dir}")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Merge a PEFT adapter into its base model for a checkpoint directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        "-c",
        required=True,
        help="Path to the adapter checkpoint directory",
    )
    args = parser.parse_args()

    merge_adapters(checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
