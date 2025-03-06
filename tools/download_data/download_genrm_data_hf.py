# I've upload the Gen-RM data as a huggingface dataset. This file downloads the data to the data folder in Llama-Factory
from datasets import load_dataset, Dataset
import os

# 1. Download the dataset
dataset = load_dataset("flowingpurplecrane/genrm", "critiques")

# 2. Save it locally with your preferred structure
# For example, if you want to reorganize JSON data:
import json

# Extract the data
data_dict = dataset["train"].to_dict()  # Or whichever split you need

print(len(data_dict["inputs"]))

# # Reorganize as needed
# # Example: restructuring or filtering data
# reorganized_data = {...}  # Your restructuring logic here

# # 3. Create a new directory structure locally
# os.makedirs("new_dataset_structure", exist_ok=True)

# # 4. Save with new structure
# with open("new_dataset_structure/data.json", "w") as f:
#     json.dump(reorganized_data, f)

# # 5. Create a new dataset from the reorganized files
# new_dataset = Dataset.from_json("new_dataset_structure/data.json")

# # 6. Upload to Hugging Face with a new name or overwrite the old one
# new_dataset.push_to_hub("your-username/your-new-dataset-name")
