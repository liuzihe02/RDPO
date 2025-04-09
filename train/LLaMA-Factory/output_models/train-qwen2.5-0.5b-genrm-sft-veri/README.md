---
library_name: peft
license: other
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: train-qwen2.5-0.5b-genrm-sft-veri
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train-qwen2.5-0.5b-genrm-sft-veri

This model is a fine-tuned version of [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) on the genrm_dpo dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-06
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 2.0

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.49.0
- Pytorch 2.4.0+cu121
- Datasets 3.2.0
- Tokenizers 0.21.0