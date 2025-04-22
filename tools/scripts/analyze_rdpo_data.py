# do some data analysis post processing to see whats the sequence lengths of our dataset so we know how much to choose cutoff_len
# in math, one token is about 2.5-3.5 characters
# question + reasoning has about 4500 characters max covering most of the data
# so a good token sequence length is about 4500/2.5=1800
# we use 1536 to be nice to GPU

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the RDPO dataset
with open("../../train/LLaMA-Factory/data/data-genrm-rdpo.json", "r") as f:
    rdpo_data = json.load(f)

# Calculate lengths
char_lengths = []
word_lengths = []
combined_char_lengths = []
combined_word_lengths = []

for entry in rdpo_data:
    q = entry["question"]
    r = entry["reasoning"]
    char_lengths.append(len(r))
    word_lengths.append(len(r.split()))
    combined_char_lengths.append(len(q + r))
    combined_word_lengths.append(len((q + " " + r).split()))


# Summary stats
def print_stats(name, data):
    print(f"\n{name} (n = {len(data)})")
    print(f"Min: {min(data)}")
    print(f"Max: {max(data)}")
    print(f"Mean: {np.mean(data):.2f}")
    print(f"Median: {np.median(data):.2f}")
    print(f"95th Percentile: {np.percentile(data, 95):.2f}")


print_stats("Reasoning length (chars)", char_lengths)
print_stats("Reasoning length (words)", word_lengths)
print_stats("Combined length (question + reasoning) - chars", combined_char_lengths)
print_stats("Combined length (question + reasoning) - words", combined_word_lengths)


# Create a 2x2 subplot for all 4 histograms
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Length Distributions of RDPO Data", fontsize=14)

axs[0, 0].hist(char_lengths, bins=30, color="skyblue")
axs[0, 0].set_title("Reasoning Length (chars)")
axs[0, 0].set_xlabel("Characters")
axs[0, 0].set_ylabel("Frequency")

axs[0, 1].hist(word_lengths, bins=30, color="lightgreen")
axs[0, 1].set_title("Reasoning Length (words)")
axs[0, 1].set_xlabel("Words")
axs[0, 1].set_ylabel("Frequency")

axs[1, 0].hist(combined_char_lengths, bins=30, color="salmon")
axs[1, 0].set_title("Combined Length (chars)")
axs[1, 0].set_xlabel("Characters")
axs[1, 0].set_ylabel("Frequency")

axs[1, 1].hist(combined_word_lengths, bins=30, color="plum")
axs[1, 1].set_title("Combined Length (words)")
axs[1, 1].set_xlabel("Words")
axs[1, 1].set_ylabel("Frequency")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("length_distributions_all.png")
print("\nSaved combined histogram to length_distributions_all.png")
