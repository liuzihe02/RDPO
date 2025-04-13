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


def check_model_id_uniqueness(dataset):
    """Check if for every unique question_id, all corresponding model_output_id values are unique.
    Returns a dictionary with question_ids that have duplicate model_output_ids.

    Exploratary Data Analysis
    """
    print("Checking model_output_id uniqueness for each question_id...")

    # Create a dictionary to store model_output_ids for each question_id
    question_models = {}
    duplicate_questions = {}

    # Iterate through the dataset
    for item in dataset:
        qid = item["question_id"]
        mid = item["model_output_id"]

        # Initialize entry for this question_id if not present
        if qid not in question_models:
            question_models[qid] = []

        # Check if this model_output_id already exists for this question
        if mid in question_models[qid]:
            # Record duplicates
            if qid not in duplicate_questions:
                # record duplicate
                duplicate_questions[qid] = []
            # record this model_output_id
            duplicate_questions[qid].append(mid)

        # Add the model_output_id to the list for this question
        question_models[qid].append(mid)

    if duplicate_questions:
        print(
            f"Found {len(duplicate_questions)} questions with duplicate model_output_ids"
        )
        for idx, (qid, mids) in enumerate(duplicate_questions.items()):
            if idx <= 10:
                print(f"Question {qid} has duplicate model_output_ids: {mids}")
    else:
        print("All questions have unique model_output_ids")

    return duplicate_questions


def check_inputs_format(dataset):
    """Check if all entries in the 'inputs' column start with the expected text.

    This is to check if the questions are basically all math questions, no last letter concatenation or word sorting as in the paper

    Exploratary Data Analysis
    """
    expected_prefix = """Solve the math problems and provide step-by-step solutions, ending with \"The answer is [Insert Final Answer Here]\".\nWhen asked \"Verification: Is the answer correct (Yes/No)?\", respond with \" Yes\" or \" No\" based on the answer's correctness.\nWhen asked \"Verification: Let's verify step by step.\", verify every step of the solution and conclude with \"Verification: Is the answer correct (Yes/No)?\" followed by \" Yes\" or \" No\"."""

    # Count total entries and those with the correct prefix
    total_entries = 0
    correct_format = 0
    incorrect_entries = []

    # Iterate through the dataset
    for i, item in enumerate(dataset):
        total_entries += 1
        inputs_text = item.get("inputs", "")

        if inputs_text and inputs_text.startswith(expected_prefix):
            correct_format += 1
        else:
            # Record the index of incorrect entries (limited to first 10)
            if len(incorrect_entries) < 10:
                incorrect_entries.append(
                    (i, inputs_text[:100] if inputs_text else "Empty")
                )

    # Print results
    print(f"\nInput format check:")
    print(f"Expected prefix: '{expected_prefix}'")
    print(f"Total entries: {total_entries}")
    print(
        f"Entries with correct format: {correct_format} ({correct_format / total_entries * 100:.2f}%)"
    )

    if incorrect_entries:
        print(f"First {len(incorrect_entries)} incorrect entries:")
        for idx, text in incorrect_entries:
            print(f"  Index {idx}: '{text}...'")

    return correct_format == total_entries


def check_verification(dataset):
    """
    Simple check:
    - If is_correct(targets) is "Yes", then "The answer is X" in the model output solution matches target value
    - If is_correct(targets) is "No", then "The answer is X" in model output solution does NOT match target value

    basically check if the verification rationales are consistent with the ground truth vs model solutions

    exploratory data analysis
    """
    print("checking verification rationales")

    inconsistencies = 0

    for item in dataset:
        # Extract from verification if the solution is marked as correct
        correct = extract_verification(item.get("targets", ""))

        # Extract the answer from inputs text
        solution = extract_solution(item["inputs"])
        answer = extract_answer(solution)
        if answer is None:
            raise ValueError(
                f"Answer does not exist! \n Inputs (raw string): {repr(item['inputs'])}, \n Target: {repr(item['target'])}, \n Marked: {correct}, \n question_id: {item['question_id']}, \n model_output_id: {item['model_output_id']}, \n Extracted Answer: {answer}, \n Extracted Solution: {repr(solution)}"
            )

        target = str(item.get("target", "")).strip()

        # Check for inconsistency
        if (correct == "Yes" and answer != target) or (
            correct == "No" and answer == target
        ):
            inconsistencies += 1
            if inconsistencies <= 10:
                print(
                    f"Inconsistency! Extracted Answer: '{answer}', Target: '{target}', Marked: {correct}, Extracted Solution: '{extract_solution(item['inputs'])}', question_id: {item['question_id']}, model_output_id: {item['model_output_id']}"
                )

    if inconsistencies == 0:
        print("All answer verification are consistent.")
    else:
        print(
            f"Found {inconsistencies} inconsistencies between answers and verifications"
        )

    return inconsistencies == 0


def extract_problem(inputs_text):
    """Extract the problem from the inputs column, including '\nQ:' but excluding '\nA:'."""
    # re.DOTALL makes the . include evrything including newline
    # . matches any character, * means zero or more, ? makes it non greedy so it stops at the first occurence of what follows
    match = re.search(r"(\nQ:.*?)\nA:", inputs_text, re.DOTALL)
    if match:
        # group 1 here to just get the matching group instead of the whole thing
        return match.group(1).strip()
    return None


def extract_solution(inputs_text):
    """Extract the full solution from the input text. In between A: xxx \nVerification"""
    solution = re.search(r"\nA:(.*?)\nVerification", inputs_text, re.DOTALL)
    if solution:
        return solution.group(1).strip()
    return None


def extract_answer(solution_text):
    """Extract the numerical final answer from a solution text. this will be a string, even though numerical"""

    # List of regex patterns to try, in order of preference
    patterns = [
        # Pattern 1: "Answer is X" or "The answer is X"
        r"answer(?:\s+is)?[:,]?\s*(\d+)",
        # Pattern 2: "The answer is X" with various endings
        r"the answer is[:,]? (.+?)(?:\.|\n|\\n|$)",
        # Pattern 3: Any numerical answer after "answer"
        r"answer.*?(\d+)",
        # Pattern 4: Any word or number after "The answer is"
        r"the\s+answer(?:\s+is)?[:,]?\s*(\S+)",
        # Pattern 5: Everything between "answer is" and a period
        r"answer\s+is\s+([^\.]+)",
        # Pattern 6: Special format like "THE ANSWER IS 588"
        r"the answer is (\d+)",
    ]
    # Try each pattern in sequence
    for i, pattern in enumerate(patterns):
        # ignore case of letters
        match = re.search(pattern, solution_text, re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            return result

    # If no patterns matched
    return None


def extract_verification(targets_text):
    """Check if the solution is correct purely based on the target text (which is verification rationale)

    basically just extract the final conclusion of the verification"""
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

target [str]: 17
"""

"""
Data processing for all files
1. Create an column called "correct", that is either "Yes" when the verification in the targets in correct is Yes, or "No". The "Yes" or "No" is end of the string. Assuming all the verifications are correct.
2. Create a "problem" string column, which is found between "\nQ:" (inclusive) and "\nA:" (exclusive)
3. Create a "solution" string column, which is found between "\nA:" (inclusive) and "\nVerification" (exclusive)
4. Sample a subset of this dataset using the cli arguments

# from all the unique questions, sample a subset of unique questions
# for each question, select a correct output and an incorrect output
# double check for each question there is EXACTLY ONE correct and one incorrect output, and that all questions are unique
#so the first row is question 1 correct solution/answer
#second row is question 1 incorrect solution/answer

Save this master dataset as data_genrm_master.json
"""


def process_master_dataset(dataset, num_samples):
    """Process the raw dataset into a master dataset with all required columns.
    First filter to include only questions with both correct and incorrect solutions,
    then sample from those eligible questions."""
    print("Processing master dataset...")

    # Convert to pandas for easier processing
    df = pd.DataFrame(dataset)

    # filter away values where target is null
    df.dropna(subset=["target"])

    # the instruction is the same for everyone

    # Extract required fields
    tqdm.pandas(desc="Extracting data")
    df["correct"] = df["targets"].progress_apply(extract_verification)
    df["problem"] = df["inputs"].progress_apply(extract_problem)
    df["solution"] = df["inputs"].progress_apply(extract_solution)
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
            # the math problem
            "problem",
            # the output solution
            "solution",
            # target is the expected answer here. a single number as a string
            "target",
            # Yes or no wheher or not correct
            "correct",
            # targets is the verification rationale here
            "targets",
        ]
    ]

    # Count rows where "targets" is null
    num_null = df["targets"].isnull().sum()
    # Count rows where "target" is an empty string (after stripping whitespace)
    num_empty = (
        df["targets"].apply(lambda x: isinstance(x, str) and x.strip() == "")
    ).sum()
    print(f"Rows with null 'targets': {num_null}")
    print(f"Rows with empty 'targets': {num_empty}")

    # First find all questions that have both correct and incorrect solutions
    print("Finding questions with both correct and incorrect solutions...")

    # Group by question_id and count occurrences of Yes and No
    question_counts = (
        df.groupby(["question_id", "correct"]).size().unstack(fill_value=0)
    )

    # Filter for questions that have at least one 'Yes' and one 'No'
    eligible_questions = question_counts[
        (question_counts["Yes"] > 0) & (question_counts["No"] > 0)
    ].index.tolist()

    print(
        f"Found {len(eligible_questions)} questions with both correct and incorrect solutions"
    )

    # Sample from eligible questions if needed, up till num samples
    if num_samples and num_samples < len(eligible_questions):
        sampled_questions = random.sample(eligible_questions, num_samples)
        print(
            f"Sampled {num_samples} questions from {len(eligible_questions)} eligible questions"
        )
    else:
        sampled_questions = eligible_questions
        print(f"Using all {len(eligible_questions)} eligible questions")

    # Create a filtered dataset with exactly one correct and one incorrect solution per question
    print("Selecting one correct and one incorrect solution per sampled question...")
    filtered_data = []

    for question_id in tqdm(sampled_questions, desc="Processing questions"):
        question_df = df[df["question_id"] == question_id]

        # Get ALL correct and incorrect solutions for this question
        correct_solutions = question_df[question_df["correct"] == "Yes"]
        incorrect_solutions = question_df[question_df["correct"] == "No"]

        # Take the FIRST correct and FIRST incorrect solution
        filtered_data.append(correct_solutions.iloc[0])
        filtered_data.append(incorrect_solutions.iloc[0])

    # Convert back to DataFrame
    master_df = pd.DataFrame(filtered_data)

    # Verify each question has exactly one correct and one incorrect solution
    verification = (
        master_df.groupby(["question_id", "correct"]).size().unstack(fill_value=0)
    )
    print(
        f"Final dataset contains {len(verification)} unique questions, each with one correct and one incorrect solution"
    )

    return master_df


"""
for the genrm_dpo:

"question": concatenate the first part of "inputs" "Solve ... with \"The answer is [Insert Final Answer Here]\"." with the "problem" column
"chosen": The "solution" column of the correct response
"rejected": The "solution" column of the incorrect response

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
        # just take the first one that you find a correct and first one that you find incorrect
        # since there can only be one of this per question
        correct_row = question_rows[question_rows["correct"] == "Yes"].iloc[0]
        incorrect_row = question_rows[question_rows["correct"] == "No"].iloc[0]

        dpo_entry = {
            "question": correct_row["instruction"] + correct_row["problem"],
            "chosen": correct_row["solution"],
            "rejected": incorrect_row["solution"],
        }

        dpo_data.append(dpo_entry)

    return dpo_data


"""
for the genrm_rdpo:

"question": concatenate the first part of "inputs" "Solve ... with \"The answer is [Insert Final Answer Here]\"." with the "problem" column
"chosen": The "solution" column of the correct response
"rejected": The "solution" column of the incorrect response
"reasoning": the "target" column of the correct response concatenated with the "target" column of incorrect response

save it as data_genrm_rdpo.json
"""


def create_rdpo_dataset(master_df):
    """Create the RDPO dataset which is similar to DPO but includes the verification data.
    For each question, the 'chosen' and 'rejected' fields are built by concatenating
    the solution with the verification rationale."""
    print("Creating RDPO dataset...")

    rdpo_data = []
    unique_questions = master_df["question_id"].unique()

    # for each question
    for qid in tqdm(unique_questions, desc="Creating RDPO entries"):
        question_rows = master_df[master_df["question_id"] == qid]
        assert len(question_rows) == 2
        correct_row = question_rows[question_rows["correct"] == "Yes"].iloc[0]
        incorrect_row = question_rows[question_rows["correct"] == "No"].iloc[0]

        rdpo_entry = {
            "question": """Solve the math problems and provide step-by-step solutions, ending with \"The answer is [Insert Final Answer Here]\"."""
            + correct_row["problem"],
            "chosen": correct_row["solution"],
            "rejected": incorrect_row["solution"],
            "reasoning": "\nThis is a correct solution and preferred: "
            + correct_row["solution"]
            + " \n\nHere's why this solution is correct and preferred: "
            + correct_row["targets"]
            + "\n\nThis is an incorrect solution and not preferred: "
            + incorrect_row["solution"]
            + "\n\nHere's why this solution is incorrect and not preferred: "
            + incorrect_row["targets"],
        }
        rdpo_data.append(rdpo_entry)

    print(f"Created {len(rdpo_data)} RDPO entries")
    return rdpo_data


"""
for the genrm_sft_no_veri dataset

ONLY use the CORRECT solution
"instruction": the first part of "inputs" column "Solve ... with \"The answer is [Insert Final Answer Here]\"."
"input": the "problem" column
"output": the "solution" column
"""


def create_sft_no_veri_dataset(master_df):
    """Create the SFT dataset without verification using only correct solutions."""
    print("Creating SFT dataset without verification...")

    # Filter for correct solutions only
    correct_df = master_df[master_df["correct"] == "Yes"]

    # Create the dataset entries directly
    sft_data = [
        {
            "instruction": row["instruction"],
            "input": row["problem"],
            "output": row["solution"],
        }
        for _, row in correct_df.iterrows()
    ]

    print(f"Created {len(sft_data)} SFT entries without verification")
    return sft_data


"""
for the genrm_sft_veri dataset:

"instruction": the front part of the "inputs" column "Solve the math problems and provide step-by-step solutions, ending with \"The answer is [Insert Final Answer Here]\".\nWhen asked \"Verification: Is the answer correct (Yes/No)?\", respond with \" Yes\" or \" No\" based on the answer's correctness.\nWhen asked \"Verification: Let's verify step by step.\", verify every step of the solution and conclude with \"Verification: Is the answer correct (Yes/No)?\" followed by \" Yes\" or \" No\"."

"input": the "problem" column
"output": the "solution" column, concatenated with the "targets" column
"""


def create_sft_veri_dataset(master_df):
    """Create the SFT dataset with verification using only correct solutions."""
    print("Creating SFT dataset with verification...")

    # Filter for correct solutions only
    correct_df = master_df[master_df["correct"] == "Yes"]

    # Create the dataset entries directly
    sft_data = [
        {
            # for verification stuff, the instruction is a little different
            "instruction": """Solve the math problems and provide step-by-step solutions, ending with \"The answer is [Insert Final Answer Here]\".\nWhen asked \"Verification: Is the answer correct (Yes/No)?\", respond with \" Yes\" or \" No\" based on the answer's correctness.\nWhen asked \"Verification: Let's verify step by step.\", verify every step of the solution and conclude with \"Verification: Is the answer correct (Yes/No)?\" followed by \" Yes\" or \" No\".""",
            "input": row["problem"],
            "output": row["solution"]
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

    # DO SOME EDA

    # # Check model_output_id uniqueness for each question_id
    # # NO the model_out_id is not unique for each question as multiple solutions may be the same
    # # however all the verification rationales are unique for each question/model_output pair
    # check_model_id_uniqueness(train_dataset)
    # print()

    # Check if all inputs have the expected format for math problems
    # basically check if all the questions are really math questions; no grammar or coding questions
    # YES: I HAVE INDEED VERIFIES THIS IS GSM8K DATA ONLY no last letter or word sort problems
    check_inputs_format(train_dataset)
    print()

    # check all our verifications are consistent - ANSWERS in the solution (model output solution) and groundtruth answers labels matches how the verification rationales labels them
    check_verification(train_dataset)
    print()

    # start processing the dataset

    # Process the master dataset with one correct and one incorrect solution per question
    master_df = process_master_dataset(train_dataset, args.num_samples)

    # Save the master dataset as a JSON file
    master_data = master_df.to_dict(orient="records")
    # note the number of samples here will be doubled, because each question has one correct and one incorrect solution
    master_filename = "data-genrm-master.json"
    # # optional to save the master dataset as we dont actually use this dataset for direct training
    save_dataset(master_data, master_filename, args.output)

    # Create and save the DPO dataset
    dpo_data = create_dpo_dataset(master_df)
    dpo_filename = "data-genrm-dpo.json"
    save_dataset(dpo_data, dpo_filename, args.output)

    # Create and save the RDPO dataset
    rdpo_data = create_rdpo_dataset(master_df)
    rdpo_filename = "data-genrm-rdpo.json"
    save_dataset(rdpo_data, rdpo_filename, args.output)

    # Create and save the SFT dataset without verification
    sft_no_veri_data = create_sft_no_veri_dataset(master_df)
    sft_no_veri_filename = "data-genrm-sft-no_veri.json"
    save_dataset(sft_no_veri_data, sft_no_veri_filename, args.output)

    # Create and save the SFT dataset with verification
    sft_veri_data = create_sft_veri_dataset(master_df)
    sft_veri_filename = "data-genrm-sft-veri.json"
    save_dataset(sft_veri_data, sft_veri_filename, args.output)

    # Print summary
    print("\nDataset processing complete!")


if __name__ == "__main__":
    main()
