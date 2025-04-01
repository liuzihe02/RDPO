# Reasoning DPO

This is a fork of the Critique Fine Tuning repo, made for initial experimentation

## Installing Packages
```bash
#somehow we need to install these packages manually
pip install packaging wheel torch==2.4.0
#we use no build isolation so it can access the dependencies in our venv
pip install -r requirements.txt --no-build-isolation
```

If the last step doesn't work, try doing `pip install -r requirements.txt --no-build-isolation`. Afterwards, follow the `LLaMA-Factory` [instructions](https://github.com/hiyouga/LLaMA-Factory) to download `LLaMA-Factory` manually.

## Notes

- `LlamaFactory` is stored in the `train/` subdirectory, install it in the train subdir before running anything
  - Data is stored in `LLaMA-Factory/data` and downloaded from `tools/scripts/download_data.sh`
  - Since we modify this extensively for RDPO, we need to upload the entire package as a folder
- need to modify the `data/datasets_info.json` in `LLaMa-Factory` **anytime you add new data**
  - Because we use our own copy of LlamaFactory, we may run into some issues
- `cutoff_len` and `per_device_train_batch_size` are by far largest factors controlling memory usage
  - Play around with this to just nice fit on device
- Deepspeed3 doesnt work with bitsandbytes for some reason
- Provide absolute file paths for evaluation scripts
- Decrease number of samples for validation to speed it up
  - If size of training dataset too big, also memory issues
- For debugging, use `logger.info_rank0(f"zihe logger after process {dataset_module['train_dataset']}")` instead of print. the `logger` object should already have been defined

## Experiments

We use the [GenRM](https://huggingface.co/datasets/flowingpurplecrane/genrm) dataset to run some initial experiments:

1. DPO w/o verification
   - For each question, select one correct answer and one incorrect answer. Do DPO on these questions.
2. SFT w/o verification
   - We only use the correct answers
3. SFT w/ verification
   - We only use the correct answers, but this time we also concatenate it with rationale for why its correct

We use a subset of `num_samples`, where we select a pair of correct and incorrect answer for each sample.

## Setup

This assumes you have already downloaded `LLaMA-Factory` in the `train/` directory

1. Download and Process Data
   - I have uploaded the GenRM dataset to huggingface
   - Simply navigate to `tools/scripts/evaluate.sh` and run this bash script
   - This will download the all the data, do some processing, and save it to `train/LLaMA-Factory/data`
   - Edit the `dataset_info.json` file to reflect the data structure (I should have already done this)
2. Train
   - Edit the corresponding `yaml` files in `train/scripts` to use `LLaMA-Factory` for training
   - Once done, navigate to the corresponding folder like `train/scripts/train-qwen2.5-0.5b-genrm-dpo` and run the corresponding bash script like `train/scripts/train-qwen2.5-0.5b-genrm-dpo/train.sh`
3. Eval
   - We modify the Qwen eval scripts to evaluate our checkpoints on the MATH-500 dataset
   - Run `train/validation/rdpo_validate.sh` and modify where the checkpoints are stored accordingly
   - If you face issues with `latex2sympy`, this is probably due to `antlr`. Uninstall `latex2sympy` and reinstall it again; it should use the correct version of `antlr` now.

## Modifying RDPO

Core files we'll have to mess around with and understand whats going on:

For DPO
- TRL: `trainer/dpo_trainer.py`
  - Although the `CustomDPOTrainer` subclasses the trl `DPOTrainer`, the `tokenize_row` function of `DPOTrainer` is never actually used! The `super().__init__` actually uses the trl generic `Trainer` class instead
- llamafactory: `data/loader.py`, `data/processor/pairwise.py`, `train/dpo/trainer.py`, maybe `data/collator.py`?
  - Note that although stage for dpo is `dpo`, backend it actually uses `rm` and hence the `PairwiseDatasetProcessor`
- transformers: `trainer.py`

Originally in DPO, we had:

```
"prompt": Solve the question
"chosen": Step 1 Step 2... The answer is 12
"rejected": Step 1 Step 2... The answer is 15
"reasoning": the first answer is better because...
```

Flow of data

```

---
llamafactory/train/dpo/workflow.py -> run_dpo func

   ...
   #this dataset_module is a dictionary of HF datasets {"train_dataset":Dataset,"eval_dataset":Dataset}

   #load the dataset module
   #this is after preprocessing via PairwiseDatasetProcessor
   dataset_module = get_dataset(template, model_args, data_args, training_args, stage="rm", **tokenizer_module)

   ...
   # Create data collator, which will put it into the right list form for Tensor
   # this is an important function that makes the chosen and rejected HF Dataset into one Tensor!
    data_collator = PairwiseDataCollatorWithPadding(
        template=template,
        model=model,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        **tokenizer_module,
    )
    ...
   # Initialize our Trainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )
        # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        #this may cause errors later on but whatever
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="rm"
            )

---
llamafactory/data/loader.py -> get_dataset func

def get_dataset(template, model_args, data_args, training_args, stage, tokenizer, processor=None):

    # Load and preprocess dataset

    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(data_args.eval_dataset, model_args, data_args, training_args, stage)

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(dataset, data_args, training_args, stage, template, tokenizer, processor)
        eval_dataset = _get_preprocessed_dataset(eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True)
      ...
      """
      logging here produces

      Dataset({
      features: ['chosen_input_ids', 'chosen_attention_mask', 'chosen_labels', 'rejected_input_ids', 'rejected_attention_mask', 'rejected_labels', 'images', 'videos', 'audios'],
      num_rows: 100 (number of pairs of groped datapoints)

      This is the Dataset object to be sent to DataCollator
      """
      # logger.info_rank0(f"zihe logger after process {dataset_module['train_dataset']}")

      return dataset_module
   

---
llamafactory/data/loader.py -> _get_preprocessed_dataset func

       dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    ...
        dataset = dataset.map(
        dataset_processor.preprocess_dataset,
         ...
    )
    #this is a HF dataset
    return dataset

---
llamafactory/data/loader.py -> _get_dataset_processor func
   ...
       elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
   return the dataset processor class itself
)

---
data/processor/pairwise.py -> PairwiseDatasetProcessor class

this contains the actual code that handles the processor of string from json file
very important file!
the order of messages in json file determines what is chosen and what is rejected
)
         
```

flow of how **training** in dpo works in `llamafactory/train/dpo/trainer.py`. The `PairwiseDataCollatorWithPadding` has the final say on how a tensor batch will look like.

```
trl compute_loss()
#note that this is only implemented for DPODataCollatorWithPadding
  ↓
llamafactory override get_batch_loss_metrics()
  ↓
llamafactory override concatenated_forward() → Gets logprobs for chosen/rejected responses
  ↓
llamafactory override compute_reference_log_probs() → Gets reference model logprobs
  ↓
llamafactory override compute_preference_loss() → Calculates preference loss from these logprobs
this calls the trl self.dpo_loss
  ↓
[combine with other losses, collect metrics, etc.]
```

### 1. Create RDPO Preprocessor

this is located in `data/processor/rdpo_pairwise.py`

---

# CritiqueFineTuning

This repo contains the code for [Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate](https://arxiv.org/abs/2501.17703). In this paper, we introduce Critique Fine-Tuning (CFT) - a paradigm shift in LLM training where models learn to critique rather than imitate!  

<a target="_blank" href="https://github.com/TIGER-AI-Lab/CritiqueFineTuning">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-black?style=flat&logo=github"></a>
<a target="_blank" href="https://arxiv.org/abs/2501.17703">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-green?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/CritiqueFineTuning">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/WebInstruct-CFT">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/collections/TIGER-Lab/critiquefinetuning-679b25e1528e75180f55e5c4">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<br>

## Highlights
Our fine-tuning method can achieve on par results with RL training!

<img width="1432" alt="abs" src="https://tiger-ai-lab.github.io/CritiqueFineTuning/static/images/teaser.png">


## News
- **[2025/01/30]** ⚡️ The paper, code, data, and model for CritiqueFineTuning are all available online.

## Getting Started

### Installation

1. First install LLaMA-Factory:
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

2. Install additional requirements:
pip install -r requirements.txt

### Training Steps

1. First, clone the repository and download the dataset:
```bash
git clone https://github.com/TIGER-AI-Lab/CritiqueFineTuning.git
cd tools/scripts
bash download_data.sh
```

2. Configure model paths in train/scripts/train_qwen2_5-math-7b-cft/qwen2.5-math-7b-cft-webinstruct-50k.yaml

3. Start training:
```bash
cd ../../train/scripts/train_qwen2_5-math-7b-cft
bash train.sh
```

For training the 32B model, follow a similar process but refer to the configuration in train/scripts/train_qwen2_5-32b-instruct-cft/qwen2.5-32b-cft-webinstruct-4k.yaml.

Note: In our paper experiments, we used MATH-500 as the validation set to select the final checkpoint. After training is complete, run the following commands to generate validation scores:
```bash
cd train/Validation
bash start_validate.sh
```
This will create a validation_summary.txt file containing MATH-500 scores for each checkpoint. Select the checkpoint with the highest score as your final model.

## Evaluation

Fill in the model path and evaluation result save path in tools/scripts/evaluate.sh, then run:
```bash
cd tools/scripts
bash evaluate.sh
```
Hardware may have a slight impact on evaluation results based on our testing. To fully reproduce our results, we recommend testing on A6000 GPU with CUDA 12.4 and vllm==0.6.6. For more environment details, please refer to requirements.txt


Note: Our evaluation code is modified from [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) and [MAmmoTH](https://github.com/TIGER-AI-Lab/MAmmoTH).

## Construct Critique Data

To create your own critique data, you can use our data generation script:

```bash
cd tools/self_construct_critique_data
bash run.sh
```
Simply modify the model_name parameter in run.sh to specify which model you want to use as the critique teacher. The script will generate critique data following our paper's approach.


## Citation

Cite our paper as
```
@misc{wang2025critiquefinetuninglearningcritique,
      title={Critique Fine-Tuning: Learning to Critique is More Effective than Learning to Imitate},
      author={Yubo Wang and Xiang Yue and Wenhu Chen},
      year={2025},
      eprint={2501.17703},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.17703},
}
```
