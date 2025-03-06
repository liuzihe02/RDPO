# remember to use a main function with argparse
# number of samples is an argument, default 100
# output directory of datasets is an argument, default this directory

# I've upload the Gen-RM data as a huggingface dataset. This file downloads the data to the data folder in Llama-Factory
import argparse
import json
import os
import re
from datasets import load_dataset, Dataset
import random
import pandas as pd
from tqdm import tqdm

random.seed(0)


def extract_problem(inputs_text):
    """Extract the problem from the inputs column, including '\nQ:' but excluding '\nA:'."""
    # re.DOTALL makes the . include evrything including newline
    # . matches any character, * means zero or more, ? makes it non greedy so it stops at the first occurence of what follows
    match = re.search(r"(\nQ:.*?)\nA:", inputs_text, re.DOTALL)
    if match:
        # group 1 here to just get the matching group instead of the whole thing
        return match.group(1).strip()
    return None


def extract_answer(inputs_text):
    """Extract the answer from the input text."""
    match = re.search(r"\nA:(.*?)\nVerification", inputs_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def is_correct(targets_text):
    """Check if the answer is correct based on target text."""
    return "Yes" if targets_text.strip().endswith("Yes") else "No"


def save_dataset(data, filename, output):
    """Save dataset to JSON file."""
    filepath = os.path.join(output, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} entries to {filepath}")


# load train split, and save the raw dataset

"""
format of the dataset:
inputs [str]: "Solve the math problems and provide step-by-step solutions, ending with \"The answer is [Insert Final Answer Here]\".\nWhen asked \"Verification: Is the answer correct (Yes/No)?\", respond with \" Yes\" or \" No\" based on the answer's correctness.\nWhen asked \"Verification: Let's verify step by step.\", verify every step of the solution and conclude with \"Verification: Is the answer correct (Yes/No)?\" followed by \" Yes\" or \" No\".\n\nQ: Eliza can iron a blouse in 15 minutes and a dress in 20 minutes.  If she spends 2 hours ironing blouses and 3 hours ironing dresses, how many pieces of clothes did she iron?\nA: Let's think step by step.\nEliza spends 2 hours on blouses so that means she must be able to iron 2 * 60 / 15 = 8 blouses in that time. Similarly, she must be able to iron 3 * 60 / 20 = 9 dresses in that time. So in total she must have ironed 8 + 9 = 17 pieces of clothing. The answer is 17.\nVerification: Let's verify step by step.\n"

targets [str]: "**Solution Verification:**\n\n1. Eliza spends 2 hours on blouses, which is equal to 2 * 60 = 120 minutes.\n2. She can iron a blouse in 15 minutes, so in 120 minutes, she can iron 120 / 15 = 8 blouses.\n3. Eliza spends 3 hours on dresses, which is equal to 3 * 60 = 180 minutes.\n4. She can iron a dress in 20 minutes, so in 180 minutes, she can iron 180 / 20 = 9 dresses.\n5. In total, she ironed 8 blouses + 9 dresses = 17 pieces of clothing.\n\n**Expected Answer:**\n\n1. Each dress takes 20 minutes or 1/3 of an hour. So in 3 hours, she can iron 9 dresses.\n2. Each blouse takes 15 minutes or 1/4 of an hour. So in 2 hours, she can iron 8 blouses.\n3. In total, she can iron 8 blouses + 9 dresses = 17 pieces of clothes.\n\n**Comparison:**\n\nThe Solution and the Expected Answer are identical.\n\n**Verification:** Is the answer correct (Yes/No)? Yes\n\n**Final Grade:** Verification: Is the answer correct (Yes/No)? Yes"

question_id [int64]: 6062

model_output_id [int64]: 16

target [str]: 301
"""

"""
Data processing for all files
1. Create an column called "correct", that is either "Yes" when the verification in the targets in correct is Yes, or "No". The "Yes" or "No" is end of the string
2. Create a "problem" string column, which is found between "\nQ:" (inclusive) and "\nA:" (exclusive)
3. Create a "answer" string column, which is found between "\nA:" (inclusive) and "\nVerification" (exclusive)
4. Sample a subset of this dataset using the cli arguments

# from all the unique questions, sample a subset of unique questions
# for each question, select a correct output and an incorrect output
# double check for each question there is EXACTLY ONE correct and one incorrect output, and that all questions are unique

Save this master dataset as data_genrm_master_{sample size}.json
"""


def process_master_dataset(dataset, num_samples):
    """Process the raw dataset into a master dataset with all required columns.
    First filter to include only questions with both correct and incorrect answers,
    then sample from those eligible questions."""
    print("Processing master dataset...")

    # Convert to pandas for easier processing
    df = pd.DataFrame(dataset)

    # filter away values where target is null
    df.dropna(subset=["target"])

    # the instruction is the same for everyone

    # Extract required fields
    tqdm.pandas(desc="Extracting data")
    df["correct"] = df["targets"].progress_apply(is_correct)
    df["problem"] = df["inputs"].progress_apply(extract_problem)
    df["answer"] = df["inputs"].progress_apply(extract_answer)
    # instruction is fixed. this insturction is for non-verification training
    df["instruction"] = (
        """Solve the math problems and provide step-by-step solutions, ending with \"The answer is [Insert Final Answer Here]\"."""
    )

    # Keep only necessary columns
    df = df[
        [
            "question_id",
            "model_output_id",
            "inputs",
            "instruction",
            "problem",
            "answer",
            # target is the expected answer here
            "target",
            "correct",
            # targets is the verification rationale here
            "targets",
        ]
    ]

    # First find all questions that have both correct and incorrect answers
    print("Finding questions with both correct and incorrect answers...")

    # Group by question_id and count occurrences of Yes and No
    question_counts = (
        df.groupby(["question_id", "correct"]).size().unstack(fill_value=0)
    )

    # Filter for questions that have at least one 'Yes' and one 'No'
    eligible_questions = question_counts[
        (question_counts["Yes"] > 0) & (question_counts["No"] > 0)
    ].index.tolist()

    print(
        f"Found {len(eligible_questions)} questions with both correct and incorrect answers"
    )

    # Sample from eligible questions if needed
    if num_samples and num_samples < len(eligible_questions):
        sampled_questions = random.sample(eligible_questions, num_samples)
        print(
            f"Sampled {num_samples} questions from {len(eligible_questions)} eligible questions"
        )
    else:
        sampled_questions = eligible_questions
        print(f"Using all {len(eligible_questions)} eligible questions")

    # Create a filtered dataset with exactly one correct and one incorrect answer per question
    print("Selecting one correct and one incorrect answer per sampled question...")
    filtered_data = []

    for question_id in tqdm(sampled_questions, desc="Processing questions"):
        question_df = df[df["question_id"] == question_id]

        # Get correct and incorrect answers for this question
        correct_answers = question_df[question_df["correct"] == "Yes"]
        incorrect_answers = question_df[question_df["correct"] == "No"]

        # Take the first correct and first incorrect answer
        filtered_data.append(correct_answers.iloc[0])
        filtered_data.append(incorrect_answers.iloc[0])

    # Convert back to DataFrame
    master_df = pd.DataFrame(filtered_data)

    # Verify each question has exactly one correct and one incorrect answer
    verification = (
        master_df.groupby(["question_id", "correct"]).size().unstack(fill_value=0)
    )
    print(
        f"Final dataset contains {len(verification)} unique questions, each with one correct and one incorrect answer"
    )

    return master_df


"""
for the genrm_dpo:

"question": concatenate the first part of "inputs" "Solve ... with \"The answer is [Insert Final Answer Here]\"." with the "problem" column
"chosen": The "answer" column of the correct response
"rejected": The "answer" column of the incorrect response

save it as data_genrm_dpo.json
"""


def create_dpo_dataset(master_df):
    """Create the DPO dataset from the master dataset."""
    print("Creating DPO dataset...")

    # Since master_df already contains exactly one correct and one incorrect per question,
    # we can process it more efficiently
    dpo_data = []

    # Get all unique question IDs
    unique_questions = master_df["question_id"].unique()

    for qid in tqdm(unique_questions, desc="Creating DPO entries"):
        # Get the pair of rows for this question
        question_rows = master_df[master_df["question_id"] == qid]

        # Extract correct and incorrect rows
        correct_row = question_rows[question_rows["correct"] == "Yes"].iloc[0]
        incorrect_row = question_rows[question_rows["correct"] == "No"].iloc[0]

        dpo_entry = {
            "question": correct_row["instruction"] + correct_row["problem"],
            "chosen": correct_row["answer"],
            "rejected": incorrect_row["answer"],
        }

        dpo_data.append(dpo_entry)

    return dpo_data


"""
for the genrm_sft_no_veri dataset

ONLY use the CORRECT answer
"instruction": the first part of "inputs" column "Solve ... with \"The answer is [Insert Final Answer Here]\"."
"input": the "problem" column
"output": the "answer" column
"""


def create_sft_no_veri_dataset(master_df):
    """Create the SFT dataset without verification using only correct answers."""
    print("Creating SFT dataset without verification...")

    # Filter for correct answers only
    correct_df = master_df[master_df["correct"] == "Yes"]

    # Create the dataset entries directly
    sft_data = [
        {
            "instruction": row["instruction"],
            "input": row["problem"],
            "output": row["answer"],
        }
        for _, row in correct_df.iterrows()
    ]

    print(f"Created {len(sft_data)} SFT entries without verification")
    return sft_data


"""
for the genrm_sft_veri dataset:

"instruction": the front part of the "inputs" column "Solve the math problems and provide step-by-step solutions, ending with \"The answer is [Insert Final Answer Here]\".\nWhen asked \"Verification: Is the answer correct (Yes/No)?\", respond with \" Yes\" or \" No\" based on the answer's correctness.\nWhen asked \"Verification: Let's verify step by step.\", verify every step of the solution and conclude with \"Verification: Is the answer correct (Yes/No)?\" followed by \" Yes\" or \" No\"."

"input": the "problem" column
"output": the "answer" column, concatenated with the "targets" column
"""


def create_sft_veri_dataset(master_df):
    """Create the SFT dataset with verification using only correct answers."""
    print("Creating SFT dataset with verification...")

    # Filter for correct answers only
    correct_df = master_df[master_df["correct"] == "Yes"]

    # Create the dataset entries directly
    sft_data = [
        {
            # for verification stuff, the instruction is a little different
            "instruction": """Solve the math problems and provide step-by-step solutions, ending with \"The answer is [Insert Final Answer Here]\".\nWhen asked \"Verification: Is the answer correct (Yes/No)?\", respond with \" Yes\" or \" No\" based on the answer's correctness.\nWhen asked \"Verification: Let's verify step by step.\", verify every step of the solution and conclude with \"Verification: Is the answer correct (Yes/No)?\" followed by \" Yes\" or \" No\".""",
            "input": row["problem"],
            "output": row["answer"]
            + "\nVerification: Let's verify step by step.\n"
            + row["targets"],
        }
        for _, row in correct_df.iterrows()
    ]

    print(f"Created {len(sft_data)} SFT entries with verification")
    return sft_data


def main():
    parser = argparse.ArgumentParser(description="Process GenRM dataset")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of unique questions to sample (default: 100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory for datasets (default: current directory)",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Download the dataset
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("flowingpurplecrane/genrm", "critiques")
    train_dataset = dataset["train"]

    # Process the master dataset with one correct and one incorrect answer per question
    master_df = process_master_dataset(train_dataset, args.num_samples)

    # Save the master dataset as a JSON file
    master_data = master_df.to_dict(orient="records")
    # note the number of samples here will be doubled, because each question has one correct and one incorrect answer
    master_filename = f"data_genrm_master_{args.num_samples * 2}.json"
    # # optional to save the master dataset as we dont actually use this dataset for direct training
    # save_dataset(master_data, master_filename, args.output)

    # Create and save the DPO dataset
    dpo_data = create_dpo_dataset(master_df)
    dpo_filename = f"data_genrm_dpo_{args.num_samples}.json"
    save_dataset(dpo_data, dpo_filename, args.output)

    # Create and save the SFT dataset without verification
    sft_no_veri_data = create_sft_no_veri_dataset(master_df)
    sft_no_veri_filename = f"data_genrm_sft_no_veri_{args.num_samples}.json"
    save_dataset(sft_no_veri_data, sft_no_veri_filename, args.output)

    # Create and save the SFT dataset with verification
    sft_veri_data = create_sft_veri_dataset(master_df)
    sft_veri_filename = f"data_genrm_sft_veri_{args.num_samples}.json"
    save_dataset(sft_veri_data, sft_veri_filename, args.output)

    # Print summary
    print("\nDataset processing complete!")
    print(
        f"- Created master dataset with {len(master_df['question_id'].unique())} unique questions"
    )
    print(f"- Created DPO dataset with {len(dpo_data)} examples")
    print(
        f"- Created SFT without verification dataset with {len(sft_no_veri_data)} examples"
    )
    print(f"- Created SFT with verification dataset with {len(sft_veri_data)} examples")
    print(f"\nAll files saved to: {args.output}")


if __name__ == "__main__":
    main()
